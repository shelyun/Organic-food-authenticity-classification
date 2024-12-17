#get data matrix using the 28 m/z pairs

import pandas as pd
import os
root=os.getcwd()
def get_file_category_columns():
    #get two columns(origin_file_name,category) in the matrix,will be used to concat with df_matrix
    df=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None,index_col=None,low_memory=False)
    df2=df.iloc[0:2,1:]
    df2=df2.T
    df2.columns=['origin_file_name','category']
    df2.loc[df2.category=='Organic','category']='1'
    df2.loc[df2.category=='NonOrganic','category']='0'
    return df2

def get_mz_pair_matrix():
    data=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None, index_col=0, skiprows=[0,1])
    data=data.T #the length of column is 16227, from 150.0138 to 1632.91
    cols_list=['151.0478_185.042',
                '151.0478_210.076',
                '151.0478_230.99',
                '151.0478_257.074',
                '151.0478_311.116',
                '159.0628_230.99',
                '159.0628_237.0748',
                '159.0628_249.038',
                '159.0628_257.074',
                '159.0628_287.137',
                '160.0756_230.99',
                '160.0756_237.0748',
                '160.0756_249.037',
                '160.0756_251.1622',
                '160.0756_406.132',
                '189.1597_210.076',
                '189.1597_237.0748',
                '189.1602_311.116',
                '192.0767_287.137',
                '210.076_257.074',
                '230.99_237.0748',
                '230.99_406.132',
                '230.991_251.1622',
                '237.0748_311.116',
                '249.037_311.116',
                '249.038_257.074',
                '251.1622_311.116',
                '287.137_406.132']
    df_matrix=pd.DataFrame(columns=cols_list)
    for pair in cols_list:
        feature1=float(pair.split('_')[0]); feature2=float(pair.split('_')[1])
        df_matrix[pair]=data[feature1]+data[feature2]

    
    #df2 is the file_name and category column dataframe
    df2=get_file_category_columns()
    df_matrix=pd.concat([df2,df_matrix],axis=1)
    df_matrix.columns=['origin_file_name','category']+cols_list
    df_matrix.to_csv(os.path.join(root,'files','compare_with_literature','mz_pair_data_matrix.csv'),index=None)
# get_mz_pair_matrix() #output a file

def normalize(df_matrix1):
    #normalization by divide the max intensity
    max_intensity=df_matrix1.stack().max()
    df_matrix2=df_matrix1.div(max_intensity)
    return df_matrix2

def get_mz_pair_matrix_normalized(): #first normalize the whole data matrix, then get m/z pair matrix
    data=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None, index_col=0, skiprows=[0,1])
    data=data.T #the length of column is 16227, from 150.0138 to 1632.91
    data=normalize(data)
    cols_list=['151.0478_185.042',
                '151.0478_210.076',
                '151.0478_230.99',
                '151.0478_257.074',
                '151.0478_311.116',
                '159.0628_230.99',
                '159.0628_237.0748',
                '159.0628_249.038',
                '159.0628_257.074',
                '159.0628_287.137',
                '160.0756_230.99',
                '160.0756_237.0748',
                '160.0756_249.037',
                '160.0756_251.1622',
                '160.0756_406.132',
                '189.1597_210.076',
                '189.1597_237.0748',
                '189.1602_311.116',
                '192.0767_287.137',
                '210.076_257.074',
                '230.99_237.0748',
                '230.99_406.132',
                '230.991_251.1622',
                '237.0748_311.116',
                '249.037_311.116',
                '249.038_257.074',
                '251.1622_311.116',
                '287.137_406.132']
    df_matrix=pd.DataFrame(columns=cols_list)
    for pair in cols_list:
        feature1=float(pair.split('_')[0]); feature2=float(pair.split('_')[1])
        df_matrix[pair]=data[feature1]+data[feature2]

    
    #df2 is the file_name and category column dataframe
    df2=get_file_category_columns()
    df_matrix=pd.concat([df2,df_matrix],axis=1)
    df_matrix.columns=['origin_file_name','category']+cols_list
    df_matrix.to_csv(os.path.join(root,'files','compare_with_literature','mz_pair_data_matrix_normalized.csv'),index=None)

# get_mz_pair_matrix_normalized() #output a file that was processed based on normalized data

import utils
import time
clf_name='adaboost'
dimn='compare_with_literature'
f_index='normalized'
with open(os.path.join(root,'files',str(dimn),clf_name+'_'+str(dimn)+'_'+str(f_index)+'.txt'),'w') as f:
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),file=f)
params_grid=utils.get_params_grid(clf_name)
utils.clf_tuning_test(clf_name,params_grid,dimn,f_index)
