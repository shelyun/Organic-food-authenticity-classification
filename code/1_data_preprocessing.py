import numpy as np
import pandas as pd
# print(pd.__version__)
# print(np.__version__)
import json
import os
root=os.getcwd()

def get_intervals(dimn):  
    #get the intervals with [dimn] windows, return intervals m/z dataframe
    #read cherrytomato markers matrix, get all the m/z markers
    data=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None, index_col=None, skiprows=[0,1]) #,dtype={0:str}
    mz_s=data[0].values.tolist() #the first col is the mass-to-charge-ratio(m/z), turn it to list

    intensity=data.iloc[:,1:]
    # mz_s=mz_s[0:20] test the first 20 rows
    msscopelow=150; msscopebig=1700
    intervals=np.linspace(msscopelow,msscopebig,dimn+1) #get m/z nums with index of 0:401 
    x1=intervals[0:dimn];x2=intervals[1:(dimn+1)] #x1, x2 are the lower and upper of each interval
    intervals_df=pd.DataFrame([x1,x2]) #2 rows, 400 cols 
    mzs_dict={x:[] for x in range(dimn)} #will includes 400 keys, the list stores related mz_s
    for mz in mz_s:
        for i in range(dimn):
            if mz >= x1[i] and mz < x2[i]:
                mzs_dict[i].append(mz)
                break
            elif mz==1700:
                mzs_dict[dimn-1].append(mz)
    for l in mzs_dict.keys():
        ll=mzs_dict[l]
        mzs_dict[l]=str(ll)
    
    #get mzs_dict, store the inform into mzs_df, and output a csv file
    mzs_df=pd.DataFrame(mzs_dict,index=[0])
    df=pd.concat([intervals_df,mzs_df])
    df1=df.T
    df1.to_csv(os.path.join(root,'files',str(dimn),'1_interval_mz.csv'),index=None,header=None)
    
    df1.columns=[0,1,2]
    return df1

def get_dimn_matrix():
    data=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None, index_col=None, skiprows=[0,1])
    df_matrix=pd.DataFrame(columns=np.arange(1601)) 
    df1=pd.read_csv(os.path.join(root,'files',str(dimn),'1_interval_mz.csv'),header=None,index_col=None)
    # data[0]=data[0].astype(str)
    for index,row in df1.iterrows():  #format: lower mz, upper mz, mz_list
        mz_list=row[2]
        mz_list=json.loads(mz_list)
        # mz_list=mz_list.strip('[').strip(']').strip().split(',')
        # mz_list = [float(x) for x in mz_list]
        # a=data[0]
        df_matrix.loc[index]=data.loc[data[0].isin(mz_list)].sum(axis=0) # sum rows by column, generate a new row in df_matrix
    df_matrix=df_matrix.iloc[:,1:]
    df_matrix1=df_matrix.T
    return df_matrix1

def get_file_category_columns():
    #get two columns(origin_file_name,category) in the matrix,will be used to concat with df_matrix
    df=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None,index_col=None,low_memory=False)
    df2=df.iloc[0:2,1:]
    df2=df2.T
    df2.columns=['origin_file_name','category']
    df2.loc[df2.category=='Organic','category']='1'
    df2.loc[df2.category=='NonOrganic','category']='0'
    return df2

def normalize(df_matrix1):
    #normalization by divide the max intensity
    max_intensity=df_matrix1.stack().max()
    df_matrix2=df_matrix1.div(max_intensity)
    return df_matrix2

def output_non_normalized_data_matrix(df2,df_matrix1):
    #output non-normalized data matrix
    df_matrix1=pd.concat([df2,df_matrix1],axis=1)
    df_matrix1.columns=['origin_file_name','category']+list(np.arange(dimn))
    df_matrix1.to_csv(os.path.join(root,'files',str(dimn),'2_mz_matrix_'+str(dimn)+'_windows.csv'),index=None)  #data formate row×column=1600*402, columns represents spectra bins

def output_normalized_data_matrix(df2,df_matrix2):
    #output normalized data matrix
    df_matrix2=pd.concat([df2,df_matrix2],axis=1)
    df_matrix2.columns=['origin_file_name','category']+list(np.arange(dimn))
    df_matrix2.to_csv(os.path.join(root,'files',str(dimn),'2_mz_matrix_'+str(dimn)+'_windows_normalized.csv'),index=None) 

def delet_all_zero_columns(df2,df_matrix,normalize=True):
    df_matrix1=df_matrix.loc[:, (df_matrix != 0).any(axis=0)]
    df_matrix1=pd.concat([df2,df_matrix1],axis=1)
    if normalize==True:
        df_matrix1.to_csv(os.path.join(root,'files',str(dimn),'3_mz_matrix_'+str(dimn)+'_windows_normalized_delete0.csv'),index=None)
    else:
        df_matrix1.to_csv(os.path.join(root,'files',str(dimn),'3_mz_matrix_'+str(dimn)+'_windows_delete0.csv'),index=None)


dimn_list=[50,200,400,600,800]

for dimn in dimn_list:

    if not os.path.exists(os.path.join(root,'files',str(dimn))):
        os.makedirs(os.path.join(root,'files',str(dimn)))
    # df1=get_intervals(dimn)
    # get_intervals(dimn)
    df_matrix1=get_dimn_matrix()
    df_matrix2=normalize(df_matrix1)
    df2=get_file_category_columns()
    output_non_normalized_data_matrix(df2,df_matrix1)
    output_normalized_data_matrix(df2,df_matrix2)
    # delet_all_zero_columns(df2,df_matrix1,normalize=False)
    # delet_all_zero_columns(df2,df_matrix2,normalize=True)


    
