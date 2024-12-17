import matplotlib.pyplot as plt
import numpy as np

'''
# 创建示例数据
cities = ['北京', '上海', '广州', '深圳', '成都']
populations_2019 = [2154, 2418, 1451, 1303, 1634]
populations_2020 = [2200, 2450, 1470, 1310, 1650]

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# 绘制第一个柱状图
axs[0].bar(cities, populations_2019)
axs[0].set_xlabel('城市')
axs[0].set_ylabel('人口')
axs[0].set_title('2019年中国各城市人口分布')

# 绘制第二个柱状图
axs[1].bar(cities, populations_2020)
axs[1].set_xlabel('城市')
axs[1].set_ylabel('人口')
axs[1].set_title('2020年中国各城市人口分布')

plt.tight_layout()
plt.show()
'''

# x=np.array(['AUPRC','AUC','Precision','Recall','F1-score','Accuracy','MCC'])
# y_50=np.array([0.9831,0.9826,0.9477,0.9062,0.9265,0.9281,0.8571])
# y_200=np.array([0.9778,0.9728,0.9669,0.9125,0.9389,0.9406,0.8826])
# y_400=np.array([0.999,0.9988,0.9938,0.9938,0.9938,0.9938,0.9875])
# y_600=np.array([0.984,0.9796,0.9865,0.9125,0.9481,0.95,0.9025])
# y_800=np.array([0.9995,0.9995,1,0.975,0.9873,0.9875,0.9753])
# y_1000=np.array([0.9998,0.9998,1,0.9938,0.9969,0.9969,0.9938])

x=np.array(['AUROC','Precision','Recall','MCC'])
y_50=np.array([0.9826,0.9477,0.9062,0.8571])
y_200=np.array([0.9728,0.9669,0.9125,0.8826])
y_400=np.array([0.9988,0.9938,0.9938,0.9875])
y_600=np.array([0.9796,0.9865,0.9125,0.9025])
y_800=np.array([0.9995,1,0.975,0.9753])
y_1000=np.array([0.9998,1,0.9938,0.9938])

# title_list=['AUPRC','AUC','Precision','Recall','F1-score','Accuracy','MCC']
title_list=['AUROC','Precision','Recall','MCC']
x=np.array(['50','200','400','600','800','1000'])
# y_auprc=np.array([0.9831,0.9778,0.999,0.984,0.9995,0.9998])
# y_auc=np.array([0.9826,0.9728,0.9988,0.9796,0.9995,0.9998])
# y_precision=np.array([0.9477,0.9669,0.9938,0.9865,1,1])
# y_recall=np.array([0.9062,0.9125,0.9938,0.9125,0.975,0.9938])
# y_f1score=np.array([0.9265,0.9389,0.9938,0.9481,0.9873,0.9969])
# y_accuracy=np.array([0.9281,0.9406,0.9938,0.95,0.9875,0.9969])
# y_mcc=np.array([0.8571,0.8826,0.9875,0.9025,0.9753,0.9938])

# y_auprc=np.array([0.9831,0.9778,0.999,0.984,0.9995,0.9998])
y_auc=np.array([0.9826,0.9728,0.9988,0.9796,0.9995,0.9998])
y_precision=np.array([0.9477,0.9669,0.9938,0.9865,1,1])
y_recall=np.array([0.9062,0.9125,0.9938,0.9125,0.975,0.9938])
# y_f1score=np.array([0.9265,0.9389,0.9938,0.9481,0.9873,0.9969])
# y_accuracy=np.array([0.9281,0.9406,0.9938,0.95,0.9875,0.9969])
y_mcc=np.array([0.8571,0.8826,0.9875,0.9025,0.9753,0.9938])
# num_list=[y_auprc,y_auc,y_precision,y_recall,y_f1score,y_accuracy,y_mcc]
num_list=[y_auc,y_precision,y_recall,y_mcc]

c_list=['b','c','g','m','r','y','k']
fig, axs = plt.subplots(1,4,sharey='row',figsize=(6,5),dpi=100)#,fontsize=15
for i in range(4):
    axs[i].barh(x,num_list[i],height=0.6,color=c_list[i])
    axs[i].set_xlim((0.85,1))
    axs[i].set_xticks((0.9,1))
    axs[i].set_title(title_list[i],fontsize=20)
    axs[i].tick_params(axis='x', which='major', labelsize=16)  # 设置x轴刻度标签字体大小为12
    axs[i].tick_params(axis='y', which='major', labelsize=16)  # 设置y轴刻度标签字体大小为12

# fig.text(-0.04, 0.5,'m/z bins', fontsize=16,va='center', rotation='vertical')
# fig.ylabel('m/z windows',fontsize=16)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
plt.tight_layout(pad=0.2)
plt.show()
a=1

# axs[0].barh(x,y_auprc)
# # plt.barh(x,y_50)
# axs[1].barh(x,y_200)
# axs[2].barh(x,y_400)
# axs[3].barh(x,y_600)
# axs[4].barh(x,y_800)
# axs[5].barh(x,y_1000)

a=1
