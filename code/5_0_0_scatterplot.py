import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

root=os.getcwd()

#heatmaps are not used
#draw heatmap
'''
dimn_list=[50,200,400,600,800,1000]
for dimn in dimn_list:

    # data=pd.read_csv(os.path.join(root,'files',str(dimn),'2_mz_matrix_'+str(dimn)+'_windows_normalized.csv'),header=0,index_col=None)
    data=pd.read_csv(os.path.join(root,'files',str(dimn),'3_mz_matrix_'+str(dimn)+'_windows_normalized_delete0.csv'),header=0,index_col=None)
    
    data=data.iloc[:,1:]

    plt.figure(dpi=100,figsize=(10,10))
    sns.heatmap(data,yticklabels=[],xticklabels=[],cmap='jet',cbar=False)
    plt.savefig(os.path.join(root,'files',str(dimn),'heatmap_delete0.png'))
    plt.clf()
'''

#draw a heatmap with colorbar
'''
dimn=50
data=pd.read_csv(os.path.join(root,'files',str(dimn),'3_mz_matrix_'+str(dimn)+'_windows_normalized_delete0.csv'),header=0,index_col=None)

data=data.iloc[:,1:]

plt.figure(dpi=100,figsize=(10,10))
sns.heatmap(data,yticklabels=[],xticklabels=[],cmap='jet',cbar=True)
plt.savefig(os.path.join(root,'files',str(dimn),'heatmap_delete0_withcbar.png'))
'''

#prepare the datasets to draw scatterplots
#process data matrix to draw scatterplot with varying point sized and hues
# three columns, m/z window, mean_intensity, spectrum num(spectrum that has peak in this window)
#normalized intensity is not used, different category(organic) should be calculate independently

'''
dimn_list=[50,200,400,600,800,1000]

for dimn in dimn_list:

    data=pd.read_csv(os.path.join(root,'files',str(dimn),'2_mz_matrix_'+str(dimn)+'_windows.csv'),header=0,index_col=None)
    data=data.iloc[:,1:]
    data0=data.loc[data['category']==0].iloc[:,1:]
    data1=data.loc[data['category']==1].iloc[:,1:]

    df=pd.DataFrame(columns=['m/z window','organic','average intensity','spectrum num'])
    df1=pd.DataFrame({'m/z window':np.arange(dimn),'organic':np.zeros(dimn),'average intensity':data0.mean(axis=0),'spectrum num':data0.astype(bool).sum(axis=0)})
    df2=pd.DataFrame({'m/z window':np.arange(dimn),'organic':np.ones(dimn),'average intensity':data1.mean(axis=0),'spectrum num':data1.astype(bool).sum(axis=0)})

    df=pd.concat([df1,df2])
    df.to_csv(os.path.join(root,'files',str(dimn),'graph_data1_with_2_mz_matrix_'+str(dimn)+'_windows.csv'),index=None)
'''

import matplotlib.pyplot as plt
import numpy as np
import math
'''
# Fixing random state for reproducibility
np.random.seed(19680801)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
'''
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def add_bottom_cax(dimn,ax, pad, width):
    '''
    在一个ax底部追加与之等宽的cax.
    pad是cax与ax的间距,width是cax的高度.
    '''
    axpos = ax.get_position() #[0.125, 0.10999999999999999], [0.9, 0.88]
    axpos1 = ax.get_xlim() #[-2.5,51.5]
    xmin=axpos1[0];xmax=axpos1[1]
    a=(-xmin)/dimn*0.7
    b=(xmax-dimn)/dimn*0.7

    caxpos = mtransforms.Bbox.from_extents(
        axpos.x0 + a,
        axpos.y0 - pad-width,
        axpos.x1 - b,
        axpos.y0 - pad
    )
    cax = ax.figure.add_axes(caxpos)

    return cax

#draw scatter plot
dimn_list=[50,200,400,600,800,1000]
for dimn in dimn_list:
    data=pd.read_csv(os.path.join(root,'files',str(dimn),'graph_data1_with_2_mz_matrix_'+str(dimn)+'_windows.csv'),header=0,index_col=None)
    #'mz_window','category','average_intensity','spec_num'
    data['organic']=data['organic'].astype(np.int64)
    data_organic=data.loc[data['organic']==1]
    data_nonorganic=data.loc[data['organic']==0]

    fig,axes=plt.subplots()
    
    im1=axes.scatter(data_organic['m/z window'],data_organic['average intensity'],s=(data_organic['spectrum num']/800)*100,cmap='plasma',c=data_organic['m/z window'],edgecolors='k',alpha=0.5)

    axes.scatter(data_nonorganic['m/z window'],data_nonorganic['average intensity'],s=(data_nonorganic['spectrum num']/800)*100,cmap='plasma',c=data_organic['m/z window'],edgecolors=None,alpha=0.5)
    axes.set(yscale='log')
    axes.set_xlabel('m/z windows')
    axes.set_ylabel('average intensity')
    fig.subplots_adjust(top=1,bottom=0.2) #[0.125, 0.10999999999999999], [0.9, 0.88]
    
    cax=add_bottom_cax(dimn=dimn,ax=axes,pad=0.1,width=0.02)

    cmap=mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=150, vmax=1700)
    im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    locator = mpl.ticker.MultipleLocator(1700)
    formatter = mpl.ticker.StrMethodFormatter('{x:.0f}')

    cbar = fig.colorbar(
        im, cax=cax, orientation='horizontal',
        ticks=[150,1700], format=formatter
    )
    t1=axes.text(0.5, -0.18, s='m/z range', horizontalalignment='center',verticalalignment='center', transform=axes.transAxes)

    #plt.legend()has three params: handles, labels, loc
    legend_elements=[plt.scatter([0],[0],marker='o',edgecolors='black',c='grey',s=45,alpha=0.5,label='1'),
                     plt.scatter([0],[0],marker='o',c='grey',s=45,alpha=0.5,label='0')]
    l1=axes.legend(handles=legend_elements,loc=[0.85,0.47],title='organic',frameon=False)

    legend_elements1=[
        plt.scatter([0],[0],marker='o',c='k',s=150/8,label='150'),
        plt.scatter([0],[0],marker='o',c='k',s=300/8,label='300'),
        plt.scatter([0],[0],marker='o',c='k',s=450/8,label='450'),
        plt.scatter([0],[0],marker='o',c='k',s=600/8,label='600'),
        plt.scatter([0],[0],marker='o',c='k',s=750/8,label='750')
    ]
    l2=axes.legend(handles=legend_elements1,loc='upper right',title='spectrum num',frameon=False)
    axes.add_artist(l1)

    # cbar.set_label('m/z range',labelpad=0)
    fig.savefig(os.path.join(root,'files',str(dimn),'scatterplot_with_cbar_legend_'+str(dimn)+'.png'))
    plt.clf()


    
'''
#draw scatter plot, not used
dimn_list=[50,200,400,600,800,1000]
for dimn in dimn_list:
    data=pd.read_csv(os.path.join(root,'files',str(dimn),'graph_data1_with_2_mz_matrix_'+str(dimn)+'_windows.csv'),header=0,index_col=None)
    #'mz_window','category','average_intensity','spec_num'
    data['organic']=data['organic'].astype(np.int64)
    # data=data.iloc[:800,1:]
    import seaborn as sns
    sns.set_theme(style="white")
    g=sns.relplot(x="m/z window", y="average intensity", hue="m/z window", size="spectrum num",
                sizes=(20, 200), alpha=.5, palette="muted",
                height=6, data=data,edgecolors='k')
    g.set(yscale="log")
    # plt.savefig(os.path.join(root,'files',str(dimn),'scatter_plot.png'))
    plt.show()
    plt.clf()
'''



