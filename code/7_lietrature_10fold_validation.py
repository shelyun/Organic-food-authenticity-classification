import pandas as pd
import os
root=os.getcwd()
import utils
import time
import random

def get_10fold_cross_validation_data():
    cv=[] #cross_validation的五种划分的数据集
    randomstate=random.sample(range(2,100),10) #random.sample(range(2,42),5)[25, 36, 18, 7, 17]
    for i in randomstate:
        cv.append(utils.split_train_val_test_set(randomstate_test=i,test_size=0.29,use_test=True))  #???
    return cv

param_grid={'n_estimators':260}
cv=get_10fold_cross_validation_data()

with open(os.path.join(root,'files','compare_with_literature','adaboost_10fold_crossvalidation_normalized.txt'),'w') as f:
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),file=f)

for idx,(train_sx,train_sy,test_sx,test_sy) in enumerate(cv):
    model = utils.classifier('adaboost',param_grid)
    model.fit(train_sx, train_sy)
    y_pred = model.predict_proba(test_sx)[:,1]
    result_dict=utils.get_values(test_sy,y_pred)
    with open(os.path.join(root,'files','compare_with_literature','adaboost_10fold_crossvalidation_normalized.txt'),'a') as f:
        # f.write('fold'+str(idx)+':\n')
        for metric in result_dict.keys():
            f.write(metric+','+str(result_dict[metric])+' ')
        f.write('\n')