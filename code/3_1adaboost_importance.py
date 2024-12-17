from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

root=os.getcwd()

def split_train_val_test_set(randomstate=1,use_test=False,test_with_duplicate=True): #randomstate refers to the way to split train & validation set
    df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)

    X=df.iloc[:,0]; Y=df.iloc[:,1]
    #random and stratified sampling, divide the samples based on ratio 6:2:2
    tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=7,stratify=Y,shuffle=True) #the samples in the test set are fixed 
    train_x, val_x, train_y, val_y = train_test_split(tv_x, tv_y, test_size=0.25,random_state=randomstate,stratify=tv_y,shuffle=True) #change the randomstate when doing cross_validation
    if use_test == False: #if use train & validation set for cross validation, stay aside the test set
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
    
'''
params={'learning_rate': 0.7, 'n_estimators': 250}
model = AdaBoostClassifier(**params)
 
model.fit(X, y)

n=model.feature_importances_#模型的重要特征
print(n)
'''

def get_index(dimn):
    df_index=pd.read_csv(os.path.join(root,'files',str(dimn),'1_interval_mz.csv'),header=None,index_col=None)
    df_index.columns=['begin','end','mzs']
    # data[u'线损率'] = data[u'线损率'].apply(lambda x: format(x, '.2%'))


    df_index['range']=df_index['begin'].apply(lambda x: round(x,2)).astype(str) + '-' + df_index['end'].apply(lambda x: round(x,2)).astype(str)
    idx_list=np.array(df_index['range'])
    # idx_list=np.round(idx_list,2)
    return idx_list


dimn_list=[1000]
f_indices=['2','3']

params={'learning_rate': 0.7, 'n_estimators': 250}

for dimn in dimn_list:
    f_index='2'
    if f_index=='2': file='2_mz_matrix_'+str(dimn)+'_windows.csv'
    if f_index=='3': file='3_mz_matrix_'+str(dimn)+'_windows_delete0.csv'
    df_matrix=pd.read_csv(os.path.join(root,'files',str(dimn),file),header=0,index_col=None)
    insert_index=get_index(dimn)
    df_matrix.columns=['origin_file_name','category']+list(insert_index)
    train_x,train_y,test_x,test_y=split_train_val_test_set(use_test=True)
    model=AdaBoostClassifier(**params)
    model.fit(train_x,train_y)
    shap_values=shap.TreeExplainer(model).shap_values(train_x)
    plt.figure(figsize=(8,10))
    # shap.summary_plot(shap_values[0],train_x,plot_type='bar')

    shap.summary_plot(shap_values[0],train_x,plot_type='dot')#violin

    '''
    n=model.feature_importances_
    importance_df=pd.DataFrame(n,columns=['importance_value'])

    insert_index=get_index(dimn)

    importance_df.insert(0, column='range',value=insert_index)
    importance_df.sort_values(by=['importance_value'],ascending=False,inplace=True)
    importance_df.to_csv(os.path.join(root,'files',str(dimn),'adaboost_importance_value'+str(dimn)+'.csv'),index=None)
    '''






