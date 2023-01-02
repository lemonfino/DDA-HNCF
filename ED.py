import pandas as pd
import numpy as np

#计算所有正样本特征向量每个维度的平均值，形成一个长度为900维的向量，并以此作为聚类中心。
train_data=pd.read_csv('pos_feat.csv',delimiter=',',header=None)
train_data=np.array(train_data)
y=np.mean(train_data,axis=0)   #聚类中心
print(y)

#计算所有未标记样本与聚类中心的欧氏距离并计算平均欧式距离
test_data1=pd.read_csv('neg_feat1.csv',delimiter=',',header=None)
test_data1=np.array(test_data1)
distance1=np.linalg.norm(test_data1-y,axis=1)
distance1 = pd.DataFrame(distance1)

test_data2=pd.read_csv('neg_feat2.csv',delimiter=',',header=None)
test_data2=np.array(test_data2)
distance2=np.linalg.norm(test_data2-y,axis=1)
distance2 = pd.DataFrame(distance2)

test_data3=pd.read_csv('neg_feat3.csv',delimiter=',',header=None)
test_data3=np.array(test_data3)
distance3=np.linalg.norm(test_data3-y,axis=1)
distance3 = pd.DataFrame(distance3)

test_data4=pd.read_csv('neg_feat4.csv',delimiter=',',header=None)
test_data4=np.array(test_data4)
distance4=np.linalg.norm(test_data4-y,axis=1)
distance4 = pd.DataFrame(distance4)

test_data5=pd.read_csv('neg_feat5.csv',delimiter=',',header=None)
test_data5=np.array(test_data5)
distance5=np.linalg.norm(test_data5-y,axis=1)
distance5 = pd.DataFrame(distance5)

test_data6=pd.read_csv('neg_feat6.csv',delimiter=',',header=None)
test_data6=np.array(test_data6)
distance6=np.linalg.norm(test_data6-y,axis=1)
distance6 = pd.DataFrame(distance6)

test_data7=pd.read_csv('neg_feat7.csv',delimiter=',',header=None)
test_data7=np.array(test_data7)
distance7=np.linalg.norm(test_data7-y,axis=1)
distance7=pd.DataFrame(distance7)

test_data8=pd.read_csv('neg_feat8.csv',delimiter=',',header=None)
test_data8=np.array(test_data8)
distance8=np.linalg.norm(test_data8-y,axis=1)
distance8 = pd.DataFrame(distance8)


all_distance1 = pd.concat([distance1,distance2,distance3,distance4,distance5,distance6,distance7,distance8])

all_distance=np.array(all_distance1)
print(all_distance)

mean_distance = np.mean(all_distance)  #平均欧氏距离 AED
print(mean_distance)

AED = mean_distance

#设定阈值，当未标记的样本与聚类中心的距离大于阈值时，则将该样本看作可靠的负样本
test_position=pd.read_csv('neg_feat_position.csv',delimiter=',',header=None)
test_position=np.array(test_position)
print(test_position)

All_data = np.concatenate((all_distance,test_position),axis=1)
print(All_data)
print(All_data.shape)

All_Data = pd.DataFrame(All_data)
All_Data.to_csv('D:/DDA-HNCF/ED and CF/2 AED/All_Data.csv',sep=',',index=False,header=False)


AED1 = All_Data[All_Data[0]>0.8*AED]    #0.14555为AED值
AED1.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED1.csv',sep=',',index=False,header=False)
print(AED1.shape)

AED2 = All_Data[All_Data[0]>0.9*AED]
AED2.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED2.csv',sep=',',index=False,header=False)
print(AED2.shape)

AED3 = All_Data[All_Data[0]>1.0*AED]
AED3.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED3.csv',sep=',',index=False,header=False)
print(AED3.shape)

AED4 = All_Data[All_Data[0]>1.1*AED]
AED4.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED4.csv',sep=',',index=False,header=False)
print(AED4.shape)

AED5 = All_Data[All_Data[0]>1.2*AED]
AED5.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED5.csv',sep=',',index=False,header=False)
print(AED5.shape)

AED6 = All_Data[All_Data[0]>1.3*AED]
AED6.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED6.csv',sep=',',index=False,header=False)
print(AED6.shape)

AED7 = All_Data[All_Data[0]>1.4*AED]
AED7.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED7.csv',sep=',',index=False,header=False)
print(AED7.shape)

AED8 = All_Data[All_Data[0]>1.5*AED]
AED8.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED8.csv',sep=',',index=False,header=False)
print(AED8.shape)

AED9 = All_Data[All_Data[0]>1.6*AED]
AED9.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED9.csv',sep=',',index=False,header=False)
print(AED9.shape)

AED10 = All_Data[All_Data[0]>1.7*AED]
AED10.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED10.csv',sep=',',index=False,header=False)
print(AED10.shape)

AED11 = All_Data[All_Data[0]>1.8*AED]
AED11.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED11.csv',sep=',',index=False,header=False)
print(AED11.shape)

AED12 = All_Data[All_Data[0]>1.9*AED]
AED12.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED12.csv',sep=',',index=False,header=False)
print(AED12.shape)

AED13 = All_Data[All_Data[0]>2.0*AED]
AED13.to_csv('D:/DDA-HNCF/ED and CF/2 AED/AED13.csv',sep=',',index=False,header=False)
print(AED13.shape)
