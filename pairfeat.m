%相互作用
interaction = load('association_matrix.txt');  
% load embedding features
drug_feature1 = load('drug_embfeat.txt');                     
dis_feature1 = load('dis_embfeat.txt');                
drug_feature2 = load('drugfeature1.txt');                     
dis_feature2 = load('disfeature1.txt');                
drug_feat = [drug_feature1,drug_feature2];                     
dis_feat = [dis_feature1,dis_feature2];                

%正样本
Pint = find(interaction);                                %%已知关联所处的位置索引     正样本
[I, J] = ind2sub(size(interaction), Pint);     %%行（药物），列（疾病）  将线性索引转换为下标
B=drug_feat(I,:);
C=dis_feat(J,:);
pos_feat=[B,C];
csvwrite('pos_feat.csv',pos_feat)
clear I J B C Pint

%总负样本
Pnoint = find(~interaction);                             %%未知关联所处的位置索引     总负样本
[k,z] = ind2sub(size(interaction), Pnoint);       %%测试集的行（药物），列（蛋白）  将线性索引转换为下标
kz =[k,z];
csvwrite('neg_feat_position.csv',kz)

kz1 =kz(1:600000,:);
D1 = drug_feat(kz1(:,1),:);
F1 = dis_feat(kz1(:,2),:);
neg_feat1=[D1,F1]; 
csvwrite('neg_feat1.csv',neg_feat1)

kz2 =kz(600001:1200000,:);
D2 = drug_feat(kz2(:,1),:);
F2 = dis_feat(kz2(:,2),:);
neg_feat2=[D2,F2];
csvwrite('neg_feat2.csv',neg_feat2)

kz3 =kz(1200001:1800000,:);
D3 = drug_feat(kz3(:,1),:);
F3 = dis_feat(kz3(:,2),:);
neg_feat3=[D3,F3];
csvwrite('neg_feat3.csv',neg_feat3)

kz4 =kz(1800001:2400000,:);
D4 = drug_feat(kz4(:,1),:);
F4 = dis_feat(kz4(:,2),:);
neg_feat4=[D4,F4];
csvwrite('neg_feat4.csv',neg_feat4)

kz5 =kz(2400001:3000000,:);
D5 = drug_feat(kz5(:,1),:);
F5 = dis_feat(kz5(:,2),:);
neg_feat5=[D5,F5];
csvwrite('neg_feat5.csv',neg_feat5)

kz6 =kz(3000001:3600000,:);
D6 = drug_feat(kz6(:,1),:);
F6 = dis_feat(kz6(:,2),:);
neg_feat6=[D6,F6];
csvwrite('neg_feat6.csv',neg_feat6)

kz7 =kz(3600001:4200000,:);
D7 = drug_feat(kz7(:,1),:);
F7 = dis_feat(kz7(:,2),:);
neg_feat7=[D7,F7];
csvwrite('neg_feat7.csv',neg_feat7)

kz8 =kz(4200001:end,:);
D8 = drug_feat(kz8(:,1),:);
F8 = dis_feat(kz8(:,2),:);
neg_feat8=[D8,F8];
csvwrite('neg_feat8.csv',neg_feat8)
