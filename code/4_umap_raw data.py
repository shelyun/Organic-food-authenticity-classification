import matplotlib.pyplot as plt
import os
import pandas as pd
from umap import UMAP
import numpy as np
from sklearn.preprocessing import StandardScaler

root=r"E:\学习文件\danish thesis\00_danish_thesis"

from sklearn.model_selection import train_test_split
def split_train_val_test_set(randomstate=1,use_test=False,test_with_duplicate=True): #randomstate refers to the way to split train & validation set
    global df,df_matrix
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
  

df=pd.read_csv(os.path.join(root,r"files\sample_label_list.csv"),header=0,index_col=None)

X=df.iloc[:,0]; Y=df.iloc[:,1]
tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=7,stratify=Y,shuffle=True)

df_matrix=pd.read_excel(os.path.join(root,'files','markers_matrix.xlsx'),header=0,index_col=None)

ltv=list(tv_x)
df_tv= df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(ltv)]
tv_sx=df_tv.iloc[:,2:]
tv_sy=df_tv.iloc[:,1]
ltest=list(test_x)
df_test=df_matrix.loc[df_matrix['origin_file_name'].map(lambda x:x.split('_'+x.split('_')[-1])[0]).isin(ltest)]
test_sx=df_test.iloc[:,2:]
test_sy=df_test.iloc[:,1]

color_list=['blue','red']
dataset_list=['train','test']

index=0
plt.figure(dpi=100,figsize=(6,5))
for data in (tv_sx,test_sx):
    reducer=UMAP(n_components=2,random_state=7)
    scaled_data=StandardScaler().fit_transform(data)
    embedding=reducer.fit_transform(scaled_data)
    plt.scatter(
        embedding[:,0],
        embedding[:,1],
        marker='o',
        s=15,
        c=color_list[index],
        edgecolors=color_list[index],
        linewidths=0.2,
        alpha=0.5
        # label=dataset_list[index]
    )
    index+=1


plt.legend([],frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('UMAP1',fontdict={'weight':'normal','size':20})
plt.ylabel('UMAP2',fontdict={'weight':'normal','size':20})

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(root,'files','umap','umap_sourcedata2.png'))
a=1


# dimn_list=[50,200,400,600,800,1000]

# for dimn in dimn_list:
        
#     f='3_mz_matrix_1000_windows_delete0.csv'
#     df_matrix=pd.read_csv(os.path.join(root,'files',str(dimn),'3_mz_matrix_'+str(dimn)+'_windows_delete0.csv'),header=0,index_col=None)

#     color_list=['blue','red']
#     dataset_list=['train','test']


#     tv_sx,tv_sy,test_sx,test_sy=split_train_val_test_set(use_test=True)
#     index=0
#     plt.figure(dpi=100,figsize=(5,5))
#     for data in (tv_sx,test_sx):
#         reducer=UMAP(n_components=2)
#         scaled_data=StandardScaler().fit_transform(data)
#         embedding=reducer.fit_transform(scaled_data)
#         plt.scatter(
#             embedding[:,0],
#             embedding[:,1],
#             s=3,
#             c=color_list[index],
#             label=dataset_list[index]
#         )
#         index+=1

#     plt.legend(loc='upper left')
#     # plt.show()
#     plt.savefig(os.path.join(root,'files',str(dimn),'umap_delete0.png'))
#     plt.clf()