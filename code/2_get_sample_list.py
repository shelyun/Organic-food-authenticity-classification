import os
root=os.getcwd()

#get sample——label file
import pandas as pd
df=pd.read_csv(os.path.join(root,'files\CherryTomato_Markers_Matrix.csv'),header=None,index_col=None,low_memory=False)
df1=df.iloc[0:2,1:]
df1=df1.T
df1.columns=['origin_file_name','category']
# df1=pd.DataFrame(columns=['origin_file_name','category'])
df1.loc[df1.category=='Organic','category']='1'
df1.loc[df1.category=='NonOrganic','category']='0'

df1['origin_file_name']=df1['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0])
df2=df1.drop_duplicates(keep='first',inplace=False)
df2.to_csv(os.path.join(root,'files\sample_label_list.csv'),index=None)
