import pandas as pd
import os
root=os.getcwd()
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve,auc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
import itertools
# from imblearn import metrics


def split_train_val_test_set(randomstate=1,randomstate_test=7,test_size=0.2,use_test=False,test_with_duplicate=True): #randomstate refers to the way to split train & validation set
    df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)
    X=df.iloc[:,0]; Y=df.iloc[:,1]
    #random and stratified sampling, divide the samples based on ratio 6:2:2
    tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=test_size,random_state=randomstate_test,stratify=Y,shuffle=True) #the samples in the test set are fixed 
    train_x, val_x, train_y, val_y = train_test_split(tv_x, tv_y, test_size=0.25,random_state=randomstate,stratify=tv_y,shuffle=True) #change the randomstate when doing cross_validation
    # df_matrix=pd.read_csv(os.path.join(root,'files','compare_with_literature','mz_pair_data_matrix.csv'),index_col=None,header=0)
    df_matrix=pd.read_csv(os.path.join(root,'files','compare_with_literature','mz_pair_data_matrix_normalized.csv'),index_col=None,header=0)
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

def model_train(clf,params,train_x,train_y,test_x):
    model=classifier(clf,params)
    model.fit(train_x,train_y)
    if clf=='pls':
        ypred = model.predict(test_x)[:,0]
    else:
        ypred = model.predict_proba(test_x)[:,1]
    return ypred  

def get_cross_validation_data():
    cv=[] #cross_validation的五种划分的数据集
    randomstate=[25, 36, 18, 7, 17] #random.sample(range(2,42),5)
    for i in randomstate:
        cv.append(split_train_val_test_set(i,use_test=False))  #???
    return cv

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
    if clf=='xgboost':model=XGBClassifier(booster='gbtree',objective='binary:logistic',**params)
    return model

def iterfun(params_grid):
    values=list(params_grid.values())
    if len(params_grid)==1: fun=itertools.product(values[0])
    if len(params_grid)==2: fun=itertools.product(values[0],values[1])
    if len(params_grid)==3: fun=itertools.product(values[0],values[1],values[2])
    if len(params_grid)==4: fun=itertools.product(values[0],values[1],values[2],values[3])
    if len(params_grid)==5: fun=itertools.product(values[0],values[1],values[2],values[3],values[4])
    return fun

def cross_validate(params_grid,cv,clf,dimn,f_index=None): #return best_score, best_parameter
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

def clf_tuning_test(clf_name,params_grid,dimn,f_index=None):
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
        for metric in result_dict.keys():
            f.write(metric+','+str(result_dict[metric])+' ')



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
