path=r'boxplotdata.xlsx'
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel(path,sheet_name='Sheet2',header=0,index_col=None)
f, ax = plt.subplots(figsize=(6, 5),dpi=100)
sns.boxplot(data=df,x='ML',y='value',palette='vlag') #,width=.6 whis=[0.45,1],
sns.stripplot(data=df,x='ML',y='value', jitter=True,marker='o',size=4, color=".3")
ax.set(xlabel="")
ax.set(ylabel="")
# ax.xaxis.grid(True)
# plt.legend([],[],frameon=False)
plt.xticks(rotation=30,fontsize=15)
plt.tight_layout(pad=0.2)
plt.show()
# sns.despine(trim=True, left=True)
a=1


'''

df=pd.read_excel(r"D:\5_graphing_and_discussion\13_box+bar_plot\boxplot_python\test set.xlsx",header=0,index_col=None)

# sns.despine(top=True, right=True, left=False, bottom=False)
plt.figure(figsize=(7,6),dpi=300)

bp=sns.boxplot(y='value',x='group',data=df,hue='type',dodge=True,gap=0.2,width=0.7)
bp=sns.stripplot(y='value',x='group',data=df,jitter=True,dodge=True,marker='o',alpha=0.8,hue='type',color='grey')

handles, labels=bp.get_legend_handles_labels()
# l=plt.legend([],frameon=False)
l=plt.legend(handles[0:3],labels[0:3],loc='lower left',frameon=False) #,loc='lower right'
for handle in l.legendHandles:
    handle.set_alpha(1)
bp.set(xlabel=None,ylabel=None)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.title('AUROC & AUPRC-test set',fontsize=25)
# plt.xticks(rotation=30)
plt.tight_layout(pad=0.2) #解决显示不完整的问题
plt.savefig(r"D:\5_graphing_and_discussion\13_box+bar_plot\boxplot_python\测试集上2 有图例.png")
plt.show()

'''

'''
df=pd.read_excel(r"D:\5_graphing_and_discussion\13_box+bar_plot\boxplot_python\training_set_AUROC.xlsx",header=0,index_col=None)

# sns.set_theme(style='ticks',palette=ownplatte)#,font='Times New Roman'
# sns.despine(top=True, right=True, left=False, bottom=False)
plt.figure(figsize=(8,7),dpi=300)

bp=sns.boxplot(y='AUROC',x='assay',data=df,hue='group',dodge=True,gap=0.1)
bp=sns.stripplot(y='AUROC',x='assay',data=df,jitter=True,dodge=True,marker='o',alpha=0.8,hue='group',color='grey')

handles, labels=bp.get_legend_handles_labels()
l=plt.legend([],frameon=False)
# l=plt.legend(handles[0:3],labels[0:3])

bp.set(xlabel=None)
bp.set(ylabel=None)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.xticks(rotation=30)
plt.title('AUROC-training set',fontsize=25)
plt.tight_layout(pad=0.2) #解决显示不完整的问题
plt.savefig(r"D:\5_graphing_and_discussion\13_box+bar_plot\boxplot_python\训练集上AUROC.png")
plt.show()
'''
