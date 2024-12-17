import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pandas as pd
path=r'boxplotdata.xlsx'
df=pd.read_excel(path,sheet_name='Sheet2',header=0,index_col=None)

list_method=['AdaBoost','LR','RF','SVM_linear','SVM_poly','PLS','k-NN','naïve bayes']
list_metric=['AUC','Precision','Recall','MCC']

colors=['#F57546','#FDB86A','#FEE99A','#F7FBAE','#87D0A5','#87D0A5','#469DB4','#4E62AB']

ax = plt.subplot(projection='3d') 
for row in df.itertuples(index=None):
    method=row[0]
    metric=row[2]
    value=row[1]
    x=int(list_method.index(method))
    y=int(list_metric.index(metric))
    z=value
    color=colors[x]

    ax.bar3d(
        x,            # 每个柱的x坐标
        y,            # 每个柱的y坐标
        0,             # 每个柱的起始坐标
        dx=0.5,          # x方向的宽度
        dy=0.25,          # y方向的厚度
        dz=z,         # z方向的高度
        color=color,
        edgecolor="black",  # 黑色描边
        shade=True)   #每个柱的颜色

# 座标轴范围
# ax.set_zlim(0.4, 1)

# ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5])#
ax.set_xticklabels([])
# ax.set_yticks([0.5,1.5,2.5,3.5])
ax.set_yticklabels([])

# 设置网格颜色和粗细
#ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "color": '#ee0009'})
ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
#ax.xaxis._axinfo["grid"]['color'] = "#ee0009"
ax.xaxis._axinfo["grid"]['linestyle'] = "-"

ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
#ax.yaxis._axinfo["grid"]['color'] = "#ee0009"
ax.yaxis._axinfo["grid"]['linestyle'] = "-"

ax.zaxis._axinfo["grid"]['linewidth'] = 0.5
#ax.zaxis._axinfo["grid"]['color'] = "#ee0009"
ax.zaxis._axinfo["grid"]['linestyle'] = "-"
 
# 设置tick的颜色的粗细
ax.tick_params(axis="both", which='both',length=0,color='white',direction='out')
# ax.tick_params(axis="both", which='major', length=2.85, width=0.57, color="red")
# ax.tick_params(axis="both", which='minor', length=2.1, width=0.5, color="blue")

plt.show()
a=1

# x = np.random.randint(0,40,10)
# y = np.random.randint(0,40,10)
# z = 80 * abs(np.sin(x+y))
# ax = plt.subplot(projection='3d')  # 三维图形

# for xx, yy, zz in zip(x,y,z):
#     color = np.random.random(3)   # 随机颜色元祖
#     ax.bar3d(
#         xx,            # 每个柱的x坐标
#         yy,            # 每个柱的y坐标
#         0,             # 每个柱的起始坐标
#         dx=1,          # x方向的宽度
#         dy=1,          # y方向的厚度
#         dz=zz,         # z方向的高度
#         color=color)   #每个柱的颜色

# plt.show()
