import numpy as np
import pandas as pd
from numpy import interp
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import csv
import math
from math import e,log
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from itertools import cycle
from gcforest.gcforest import GCForest
import pickle
import os 
from scipy import interp
from tensorflow.keras import regularizers


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, errors='ignore'))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return
#读取正样本集、负样本集名字
PositiveSample = []
ReadMyCsv(PositiveSample, "D:/DDA-HNCF/ED and CF/data/PositiveSample.csv")
NegativeSample = []
ReadMyCsv(NegativeSample, "D:/DDA-HNCF/ED and CF/data/NegativeSample.csv")
Sample=PositiveSample+NegativeSample
X = np.array(Sample)   #关联对名字                 ##得到基准数据集中药物和疾病名字

#生成样本标签（正为1，负为0）
SampleLabel = []
counter = 0
while counter < len(Sample) / 2:
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(Sample) / 2:
    SampleLabel.append(0)
    counter1 = counter1 + 1      
y = np.array(SampleLabel)   #关联对标签           ##得到基准数据集中标签

#药物名字、疾病名字
AllDrug=[]
ReadMyCsv(AllDrug, "D:/DDA-HNCF/ED and CF/data/drug_num.csv")
AllDrug = np.array(AllDrug) 
AllDisease=[]
ReadMyCsv(AllDisease, "D:/DDA-HNCF/ED and CF/data/dis_num.csv")
AllDisease = np.array(AllDisease)
#读取所有药物特征、所有疾病特征
Drugfeature = np.loadtxt('D:/DDA-HNCF/ED and CF/data/drug_feature.csv', delimiter=',')
Disfeature = np.loadtxt('D:/DDA-HNCF/ED and CF/data/dis_feature.csv',delimiter=',')

#匹配样本对应的药物
X1 = pd.DataFrame(columns = np.arange(1)+1, index = np.arange(112646)+1)
count1=1
for i in range(len(X)):
    for j in range(len(AllDrug)):
        if X[i,0]==AllDrug[j,0]:
            X1.loc[count1]=j
            count1 += 1
X1 = np.array(X1)  
#匹配样本中药物对应的特征
drug_feature = pd.DataFrame(columns = np.arange(900)+1, index = np.arange(112646)+1)
count2 = 1
for i in range(len(X1)):
    drug_feature.loc[count2] = Drugfeature[X1[i,0],:]
    count2 += 1
print(drug_feature)
drug_feature = np.array(drug_feature)           ###得到基准数据集中药物特征

#匹配样本对应的疾病
X2=pd.DataFrame(columns = np.arange(1)+1, index = np.arange(112646)+1)
countr1=1
for m in range(len(X)):
    for n in range(len(AllDisease)):
        if X[m,1]==AllDisease[n,0]:
            X2.loc[countr1]=n
            countr1 += 1
X2 = np.array(X2)   
#匹配样本中疾病对应的特征
disease_feature = pd.DataFrame(columns = np.arange(900)+1, index = np.arange(112646)+1)
countr2 = 1
for i in range(len(X2)):
    disease_feature.loc[countr2] = Disfeature[X2[i,0],:]
    countr2 += 1
print(disease_feature)
disease_feature = np.array(disease_feature)     ###得到基准数据集中疾病特征

#拼接，得到药物-疾病关联对的特征 
X3 = np.concatenate((drug_feature, disease_feature), axis=1)   ####得到基准数据集中关联对特征


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 42
    ca_config["max_layers"] = 10  #最大的层数，layer对应论文中的level
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 2   #判别的类别数量
    ca_config["estimators"] = []
    
    rf_1 = {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_features": 4,
          "min_samples_split": 10, "max_depth": None, "n_jobs": -1}
    
    rf_2 = {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 300, "max_features": 6, 
          "min_samples_split": 12, "max_depth": None, "n_jobs": -1}

    xgb_1 = {"n_folds": 5, "type": "XGBClassifier", 'booster':'gbtree', 'colsample_bylevel':1,
           'colsample_bytree':1, 'eval_metric':'auc', 'gamma':0, 'learning_rate': 0.01,
           'max_delta_step':0, 'max_depth':4, 'min_child_weight':0.001,
           'n_estimators':100, 'n_jobs':-1, 'nthread':-1, 'objective':'binary:logistic', 
           'scale_pos_weight':1, 'seed':42, 'subsample':1}
    
    xgb_2 = {"n_folds": 5, "type": "XGBClassifier", 'booster':'gbtree', 'colsample_bylevel':1, 
           'colsample_bytree':1, 'eval_metric':'auc', 'gamma':0, 'learning_rate': 0.1,
           'max_delta_step':0, 'max_depth':6, 'min_child_weight':0.1,
           'n_estimators':300, 'n_jobs':-1, 'nthread':-1, 'objective':'binary:logistic', 
           'scale_pos_weight':1, 'seed':42, 'subsample':1}

    ca_config["estimators"].append(rf_1)
    ca_config["estimators"].append(rf_2)
    ca_config["estimators"].append(xgb_1)
    ca_config["estimators"].append(xgb_2)
    config["cascade"] = ca_config
    return config
def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)   #recall
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)    #precision
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F=2* Prec * Sen/ (Prec + Sen)
    # 分母可能出现0，需要讨论待续
    print('Accuracy:', Acc)
    print('Sen/recall:', Sen)
    print('Spec:', Spec)
    print('precision:', Prec)
    print('Mcc:', MCC)
    print('f1-score:', F)
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    Result.append(round(F, 4))
    return Result

#交叉验证折数
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42) # 五折交叉验证，随机种子取为60
AllResult = []
Aucs = []
Prcs = []
Fprs = []
Tprs = []
Recalls = []
Precisions = []
p=0
for train, test in cv.split(X3, y):
    print("TRAIN:", train, "TEST:", test)       
    X_train, X_test = X3[train], X3[test]
    y_train, y_test = y[train], y[test]        
    config=get_toy_config()   
    gc = GCForest(config)                
    X_train_enc = gc.fit_transform(X_train, y_train)         
    y_pred = gc.predict(X_test)  #返回预测标签    
    y_score1 = gc.predict_proba(X_test)  #返回预测属于某标签的概率    
    
    fpr, tpr, thresholds1 = roc_curve(y_test, y_score1[:, 1])  #不同阈值下的fpr和tpr
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds2 = precision_recall_curve(y_test, y_score1[:, 1])
    roc_prc = auc(recall, precision)    ##auc为pr曲线下的面积
    
    Fprs.append(fpr)
    Tprs.append(tpr)
    Precisions.append(precision)
    Recalls.append(recall)
    Aucs.append(roc_auc)     
    Prcs.append(roc_prc)     
                
    Result =MyConfusionMatrix(y[test],y_pred)
    AllResult.append(Result)
    AllResult[p].append(roc_auc)
    AllResult[p].append(roc_prc)       
    p=p+1


def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    SumF=0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        SumF = SumF + matrix[counter][5]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    print('AverageF:', SumF / len(matrix))
    return
averagevalue = MyAverage(AllResult)

# 均值
mean_auc = np.mean(Aucs, axis=0)    #计算平均AUC值
mean_prc = np.mean(Prcs, axis=0)    #计算平均AUC值

# 画平均roc曲线
plt.figure(1) # 创建图表1
plt.title('AUROC')
plt.plot(fpr, tpr, color='darkorange',label='Mean ROC(area = %0.4f)' % (mean_auc))
plt.xlim([-0.02, 1.01])
plt.ylim([0.65, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('D:/DDA-HNCF/ED and CF/mean_auc.png')
plt.show()
# 画平均prc曲线
plt.figure(2) 
plt.title('AUPRC')
plt.plot(recall, precision, label='Mean PRC(area = %0.4f)' % (mean_prc), color='darkorange')
plt.xlim([-0.02, 1.01])
plt.ylim([0.65, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.savefig('D:/DDA-HNCF/ED and CF/mean_prc.png')
plt.show()



