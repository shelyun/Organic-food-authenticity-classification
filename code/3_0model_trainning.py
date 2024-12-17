import pandas as pd
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve,auc
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import time
from math import log10
gammas = np.logspace(-6, -1, 10,base=10)
a=[log10(6), log10(5)]
#, log10(-4), log10(-3), log10(-2), log10(-1)]
degree = [2] #,3,4,5 
Cs = np.geomspace(0.0001, 1000, num=8)

root=os.getcwd()

def split_train_val_test_set(randomstate=1,use_test=False,test_with_duplicate=True): #randomstate refers to the way to split train & validation set
    df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)

    X=df.iloc[:,0]; Y=df.iloc[:,1]
    #random and stratified sampling, divide the samples based on ratio 6:2:2
    tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=7,stratify=Y,shuffle=True) #the samples in the test set are fixed 
    train_x, val_x, train_y, val_y = train_test_split(tv_x, tv_y, test_size=0.25,random_state=randomstate,stratify=tv_y,shuffle=True) #change the randomstate when doing cross_validation
    if use_test == False: #if use train & validation set for cross validation, stay the test set aside
        ltrain=list(train_x)
        df_train=df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(ltrain)] #filter the spectra that has sample name in the ltrain(train sample list)
        train_sx=df_train.iloc[:,2:]
        train_sy=df_train.iloc[:,1]
        # print(train_sy.value_counts())
        lval=list(val_x)
        df_val=df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(lval)] #filter the validation spetra
        val_sx=df_val.iloc[:,2:]
        val_sy=df_val.iloc[:,1]
        # print(val_sy.value_counts())
        return(train_sx,train_sy,val_sx,val_sy)
    elif use_test == True and test_with_duplicate == True: # when train the all train & internal validation set, and use the test set to see performance metric
        ltv=list(tv_x)
        df_tv= df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(ltv)]
        tv_sx=df_tv.iloc[:,2:]
        tv_sy=df_tv.iloc[:,1]
        ltest=list(test_x)
        df_test=df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(ltest)]
        test_sx=df_test.iloc[:,2:]
        test_sy=df_test.iloc[:,1]
        return(tv_sx,tv_sy,test_sx,test_sy)
    elif use_test == True and test_with_duplicate == False:
        ltv=list(tv_x)
        df_tv= df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(ltv)]
        tv_sx=df_tv.iloc[:,2:]
        tv_sy=df_tv.iloc[:,1]
        ltest=list(test_x)
        df_test=pd.DataFrame(columns=['category']+list(tv_sx.columns))
        for idx,sample in enumerate(ltest):
            df_test_idx=df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]==sample)].iloc[:,1:]
            df_test.loc[idx]=df_test_idx.mean(0)
        test_sx=df_test.iloc[:,1:]
        test_sy=df_test.iloc[:,0]
        return(tv_sx,tv_sy,test_sx,test_sy)
    
def objective(trial):
    param = {
        'booster':'gbtree',
        "objective": "binary:logistic",
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 50,300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        # 'gamma': trial.suggest_float('gamma', 1e-7, 10.0,log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0,log=True),
        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0,log=True),
        # 'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
    }
    cv_scores=np.zeros(5)

    for idx,(train_sx,train_sy,test_sx,test_sy) in enumerate(cv):
        #=======test if the global parameter cv is different?
        model = XGBClassifier(**param)
        model.fit(train_sx, train_sy) #, early_stopping_rounds=100
        y_pred = model.predict_proba(test_sx)[:,1]
        cv_scores[idx] = roc_auc_score(test_sy,y_pred)

    return np.mean(cv_scores)

def optuna_tuning(objective1):
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    n_trials=50 #2
    study.optimize(objective1, n_trials=n_trials)
    print('Number of finished trials:', len(study.trials))
    print("------------------------------------------------")
    print('Best trial:', study.best_trial.params)
    print("------------------------------------------------")
    params=study.best_params
    return params

def xgboost_train(train_x,train_y,test_x,params):#,test_y,model_path
    model=XGBClassifier(booster='gbtree',objective='binary:logistic',**params) 
    model.fit(train_x, train_y)
    # model.save_model(model_path) #??
    ypred = model.predict_proba(test_x)[:,1]
    return ypred

def get_values(test_y,ypred,threshold=0.5): #print test results
    y_pred = (ypred >=threshold)*1

    precision, recall, thresholds = precision_recall_curve(test_y, ypred)
    print('AUPRC: %.4F' %auc(recall,precision))
    print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))

    print ('Precision: %.4f' %metrics.precision_score(test_y,y_pred))
    print ('Recall: %.4f' %metrics.recall_score(test_y,y_pred))
    print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
    print ('Accuracy: %.4f' %metrics.accuracy_score(test_y,y_pred))
    print ('MCC: %.4f' %metrics.matthews_corrcoef(test_y,y_pred))
    result_dict={'AUPRC':'%.4F' %auc(recall,precision),
                 'AUC':'%.4f' % metrics.roc_auc_score(test_y,ypred),
                 'Precision':'%.4f' %metrics.precision_score(test_y,y_pred),
                 'Recall': '%.4f' %metrics.recall_score(test_y,y_pred),
                 'F1-score': '%.4f' %metrics.f1_score(test_y,y_pred),
                 'Accuracy': '%.4f' %metrics.accuracy_score(test_y,y_pred),
                 'MCC': '%.4f' %metrics.matthews_corrcoef(test_y,y_pred)}
    return result_dict


from sklearn import svm
def model_train(clf,params,train_x,train_y,test_x):
    model=classifier(clf,params)
    model.fit(train_x,train_y)
    if clf=='pls':
        ypred = model.predict(test_x)[:,0]
    else:
        ypred = model.predict_proba(test_x)[:,1]
    return ypred


#classifier=svm.SVC(probability=True,**params)
#classifier=XGBClassifier(booster='gbtree',objective='binary:logistic',**params) 

def get_cross_validation_data():
    cv=[] #cross_validation的五种划分的数据集
    randomstate=[25, 36, 18, 7, 17] #random.sample(range(2,42),5)
    for i in randomstate:
        cv.append(split_train_val_test_set(i,use_test=False))  #???
    return cv
    
import itertools
def svm_kernel_params(kernelidx=0):
    gammas = np.logspace(-6, -1, 10,base=10)
    degree = [2] #,3,4,5 
    Cs = np.geomspace(0.0001, 1000, num=8)
    parameters = [
    {'kernel': ['linear'], 'C': Cs},
    {'kernel': ['rbf'], 'C': Cs, 'gamma': gammas},
    {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree' : degree}, #degree means the degree of polynomial kernel function, but cost much more time when degree is more than 2
    ]
    params=parameters[kernelidx]
    return params

def classifier(clf,params):
    #clf is in ['svm','xgboost','adaboost']
    if clf[0:3]=='svm': model=svm.SVC(probability=True,**params)
    if clf=='xgboost': model=XGBClassifier(booster='gbtree',objective='binary:logistic',**params)
    if clf=='adaboost': model=AdaBoostClassifier(**params)
    if clf=='pls':model=PLSRegression(**params,scale=False)
    if clf=='random_forest':model=RandomForestClassifier(**params)
    if clf=='logistics_regression':model=LR(solver='newton-cg',**params)
    if clf=='naive_bayes':model=MultinomialNB(**params)
    if clf=='knn':model=KNeighborsClassifier(**params)
    return model

def iterfun(params_grid):
    values=list(params_grid.values())
    if len(params_grid)==1: fun=itertools.product(values[0])
    if len(params_grid)==2: fun=itertools.product(values[0],values[1])
    if len(params_grid)==3: fun=itertools.product(values[0],values[1],values[2])
    if len(params_grid)==4: fun=itertools.product(values[0],values[1],values[2],values[3])
    if len(params_grid)==5: fun=itertools.product(values[0],values[1],values[2],values[3],values[4])
    return fun

def cross_validate(params_grid,cv,clf,dimn,f_index): #return best_score, best_parameter
    #params is a dict that includes several parameters and corresponding candidates 
    #clf is in ['svm','xgboost','adaboost','random_forest']
    best_score=0.0

    fun=iterfun(params_grid)
    params_groups=[]
    for i in fun:
        params_groups.append(i)

    # def get_cv_scores(p):
    #     params_i=params_grid
    #     for idx0,value in enumerate(p):
    #         params_i[list(params_i.keys())[idx0]]=value
    #     cv_scores=np.zeros(5)
    #     for idx,(train_sx,train_sy,test_sx,test_sy) in enumerate(cv):
    #         model = classifier(clf,params_i)
    #         model.fit(train_sx, train_sy)
    #         y_pred = model.predict_proba(test_sx)[:,1]
    #         cv_scores[idx] = roc_auc_score(test_sy,y_pred)
    #     score=np.mean(cv_scores)
    #     print('score',score,',with parameter:',params_i)
    #     return score,params_i

    scores=[]
    for p in params_groups:
        params_i=params_grid
        for idx0,value in enumerate(p):
            params_i[list(params_i.keys())[idx0]]=value
        cv_scores=np.zeros(5)
        for idx,(train_sx,train_sy,test_sx,test_sy) in enumerate(cv):
            model = classifier(clf,params_i)
            model.fit(train_sx, train_sy)
            if clf=='pls':
                y_pred = model.predict(test_sx)[:,0]
            else:
                y_pred = model.predict_proba(test_sx)[:,1]
            cv_scores[idx] = roc_auc_score(test_sy,y_pred)
        score=np.mean(cv_scores)
        scores.append(score)
        with open(os.path.join(root,'files',str(dimn),clf+'_'+str(dimn)+'_'+str(f_index)+'.txt'),'a') as f:
            print('score',score,',with parameter:',params_i,file=f)
        if score>best_score:
            best_score=score
            # best_parameter=params_i
        # result.append()
    # return result
    print('best score',best_score)
    return scores,params_groups

from sklearn.cross_decomposition import PLSRegression
def pls_train(cv=None,components=None,max_components=100):
    if components==None:
        auc=[]
        for i in range(1,max_components+1):
            cv_scores=np.zeros(5)
            for idx,(train_sx,train_sy,test_sx,test_sy) in enumerate(cv):
                pls2=PLSRegression(n_components=i,scale=False)
                pls2.fit(train_sx,train_sy)
                y_pred = pls2.predict(test_sx)[:,0]
                cv_scores[idx] = roc_auc_score(test_sy,y_pred)
            score=np.mean(cv_scores)
            print('n_components ',i,',score: ',score)
            auc.append(score)
        components=auc.index(max(auc))+1
        print('best n_components:',components)
        return components
    elif components != None:
        model=PLSRegression(n_components=components,scale=False)
        return model
    
#PLS, find the best num_components and test
'''
dimn=1000
df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)
f='3_mz_matrix_1000_windows_delete0.csv'
df_matrix=pd.read_csv(os.path.join(root,'files',str(dimn),f),header=0,index_col=None)

cv=get_cross_validation_data()
components=pls_train(cv)

train_x,train_y,test_x,test_y=split_train_val_test_set(use_test=True)
model=pls_train(components=components)
model.fit(train_x,train_y)
ypred=model.predict(test_x)[:,0]
get_values(test_y,ypred)
'''

#xgboost, parameter tunning and test
#=====
'''
dimn=1000
df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)
flist=os.listdir(os.path.join(root,'files',str(dimn)))
flist.remove('1_interval_mz.csv')
# flist.remove('2_mz_matrix_1000_windows.csv')
# flist.remove('2_mz_matrix_1000_windows_normalized.csv')
flist=['3_mz_matrix_1000_windows_normalized_delete0.csv']
for i in flist:
    print(i)
    df_matrix=pd.read_csv(os.path.join(root,'files',str(dimn),i),header=0,index_col=None) #400 features and 1600 columns

    # cv = get_cross_validation_data()
    # params=optuna_tuning(objective)

    params={'max_depth': 10, 'learning_rate': 0.1281975631742061, 'n_estimators': 337, 'min_child_weight': 18, 'gamma': 0.00037870473240137804, 'subsample': 0.5886508626516654, 'colsample_bytree': 0.6747908563431476, 'reg_alpha': 0.001479737386769513, 'reg_lambda': 0.0571188216237201, 'random_state': 48}
    
    print('test set has duplicate samples')
    datasets=split_train_val_test_set(use_test=True)
    ypred=xgboost_train(datasets[0],datasets[1],datasets[2],params=params)
    get_values(datasets[3],ypred)

    print('test set does not have duplicate samples')
    datasets1=split_train_val_test_set(use_test=True,test_with_duplicate=False)
    ypred1=xgboost_train(datasets1[0],datasets1[1],datasets1[2],params=params)
    get_values(datasets[3],ypred)
#=====
'''
#svm, parameter tuning and test
#=====
'''
dimn=1000
df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)
f='3_mz_matrix_1000_windows_delete0.csv'
df_matrix=pd.read_csv(os.path.join(root,'files',str(dimn),f),header=0,index_col=None)

cv = get_cross_validation_data()
params_grid=svm_kernel_params(kernelidx=0) #kernelidx:0->linear, 1->rbf, 2->poly
scores,params_all=cross_validate(params_grid,cv,'svm') 

score=max(scores)
params_tuple=params_all[scores.index(max(scores))]
params=params_grid
for idx,value in enumerate(params_tuple):
    params[list(params.keys())[idx]]=value
print('best score',score,'with best parameter:',params)

# params= {'kernel': 'poly', 'C': 0.0001, 'gamma': 0.02782559402207126, 'degree': 2}
train_x,train_y,test_x,test_y=split_train_val_test_set(use_test=True)
ypred=model_train('svm',params,train_x,train_y,test_x)
get_values(test_y,ypred)
#=====
'''


def clf_tuning_test(clf_name,params_grid,dimn,f_index):
    cv = get_cross_validation_data()

    scores,params_all=cross_validate(params_grid,cv,clf_name,dimn,f_index)

    score=max(scores)
    params_tuple=params_all[scores.index(max(scores))]
    params=params_grid
    for idx,value in enumerate(params_tuple):
        params[list(params.keys())[idx]]=value

    with open(os.path.join(root,'files',str(dimn),clf_name+'_'+str(dimn)+'_'+str(f_index)+'.txt'),'a') as f:
        print('best score',score,'with best parameter:',params,file=f)
    

    train_x,train_y,test_x,test_y=split_train_val_test_set(use_test=True)
    ypred=model_train(clf_name,params,train_x,train_y,test_x)
    result_dict=get_values(test_y,ypred)

    with open(os.path.join(root,'files',str(dimn),clf_name+'_'+str(dimn)+'_'+str(f_index)+'.txt'),'a') as f:
        for matric in result_dict.keys():
            f.write(matric+','+str(result_dict[matric]))



def get_params_grid(clf):
    gammas = np.logspace(-6, -1, 10,base=10)
    degree = [2] #,3,4,5 
    Cs = np.geomspace(0.0001, 1000, num=8)

    clf_params={'pls':{'n_components':range(1,101,1)},
                'adaboost':{'learning_rate':[0.1,0.3,0.5,0.7,1],'n_estimators':range(150,300,50)},
                'svm_linear':{'kernel': ['linear'], 'C': Cs},
                'svm_rbf':{'kernel': ['rbf'], 'C': Cs, 'gamma': gammas},
                'svm_poly':{'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree' : degree}, #degree means the degree of polynomial kernel function, but cost much more time when degree is more than 2
                'random_forest':{'n_estimators':range(1,101,10),'max_features':range(1,11,1)},
                'logistics_regression':{'C':[1.0,1.5,2.0,2.5],'max_iter':[100,110,120]},
                'naive_bayes':{'alpha':[0.0001,0.001,0.01,0.1,1,10,100]},
                'knn':{'n_neighbors':range(1,11,1)}}
    
    params_grid=clf_params[clf]
    return params_grid


# clf_list=['pls','adaboost','svm_linear','svm_rbf','svm_poly','random_forest','logistics_regression','naive_bayes','knn']
clf_list=['pls']


# dimn_list=[50,200,400,600,800,1000]
dimn_list=[1000]
# f_indices=['2','2_normalized','3_delete0','3_normalized_delete0']
f_indices=['2','3']

# clf_list.remove('adaboost')
# clf_list.remove('pls')
for dimn in dimn_list:
    # for f_index in f_indices:
    for clf_name in clf_list:
        f_index='2'
        if f_index=='2': file='2_mz_matrix_'+str(dimn)+'_windows.csv'
        if f_index=='3': file='3_mz_matrix_'+str(dimn)+'_windows_delete0.csv'
        
        df_matrix=pd.read_csv(os.path.join(root,'files',str(dimn),file),header=0,index_col=None)

        # clf_name='adaboost'
        with open(os.path.join(root,'files',str(dimn),clf_name+'_'+str(dimn)+'_'+str(f_index)+'.txt'),'w') as f:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),file=f)

        params_grid=get_params_grid(clf_name)
        clf_tuning_test(clf_name,params_grid,dimn,f_index)
# params_grid={'alpha':[0.0001,0.001,0.01,0.1,1,10,100]}
# clf_tuning_test('naive_bayes',params_grid)

# params_grid={'n_neighbors':range(1,11,1)}
# clf_tuning_test('knn',params_grid)
'''
params_grid={"n_estimators": [50,100,150,200,300],
        "learning_rate": [0.05, 0.1, 0.2, 0.3],
        "max_depth": [3,4,5,6,7],
        "colsample_bytree": [0.4,0.6,0.8,1],
        "min_child_weight": [1,2,3,4]}

clf_tuning_test('xgboost',params_grid,dimn)
'''