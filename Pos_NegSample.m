%打开数据
interaction = load('association_matrix.txt');
% load embedding features
drug_feature1 = load('drug_embfeat.txt');                     
dis_feature1 = load('dis_embfeat.txt');                
drug_feature2 = load('drugfeature1.txt');                     
dis_feature2 = load('disfeature1.txt');                
drug_feat = [drug_feature1,drug_feature2];                     
dis_feat = [dis_feature1,dis_feature2]; 
csvwrite('D:/DDA-HNCF/ED and CF/data/drug_feature.csv',drug_feat)
csvwrite('D:/DDA-HNCF/ED and CF/data/dis_feature.csv',dis_feat)
clear drug_feature1 drug_feature2 dis_feature1 dis_feature2 drug_feat dis_feat

%筛出正样本
pos = find(interaction);
nint = length(pos);
[a1,b1] = ind2sub(size(interaction),pos);
Positive = [a1,b1];
clear pos a1 b1
%匹配正样本对应的药物、疾病名字
n =0;
for i =1:length(Positive)
    n =n+1;
    positivename(n,1) = drugnum(Positive(i,1),1);
    positivename(n,2) = disnum(Positive(i,2),1);
end
clear i n 
writematrix(positivename,'D:/DDA-HNCF/ED and CF/data/PositiveSample.csv');

%筛出负样本
AED6 = load('AED12.csv');
%Pnoint = (1:1:(length(AED7)))';
Pnoint = linspace(1,length(AED6),size(AED6,1))';
Pnoint1 = Pnoint(randperm(length(AED6), nint * 1));%随机抽取同样数量的负样本
clear  Pnoint
n =0;
for i =1:length(Pnoint1)
    n =n+1;
    k(n,1)=AED6(Pnoint1(i,1),2);
end
clear i j n
n =0;
for i =1:length(Pnoint1)
    n=n+1;
    z(n,1)=AED6(Pnoint1(i,1),3);
end
clear i j n

Negative =[k,z];
clear k z Pnoint1 nint

%匹配负样本对应的药物、疾病
n =0;
for i =1:length(Negative)
    n =n+1;
    negativename(n,1) = drugnum(Negative(i,1),1);
    negativename(n,2) = disnum(Negative(i,2),1);
end
clear i n 
writematrix(negativename,'D:/DDA-HNCF/ED and CF/data/NegativeSample.csv');

