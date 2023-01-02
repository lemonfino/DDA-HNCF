import numpy as np
import GAT_layer
from GAT_layer import *

def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))  #全0矩阵
    dis_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]),dtype=np.int8))  #疾病全0矩阵
    mat1 = np.hstack((drug_matrix,drug_dis_matrix))  #2202*3912
    mat2 = np.hstack((drug_dis_matrix.T,dis_matrix))  #1710*3912
    return np.vstack((mat1,mat2))  #3912*3912

def Net2edgelist(drug_dis_matrix_net):
    none_zero_position = np.where(np.triu(drug_dis_matrix_net) != 0)  #网络中为1的位置
    none_zero_row_index = np.mat(none_zero_position[0],dtype=int).T   #行的倒置 （网络中为1的行位置）
    none_zero_col_index = np.mat(none_zero_position[1],dtype=int).T   #列的倒置 （网络中为1的列位置）
    none_zero_position = np.hstack((none_zero_row_index,none_zero_col_index))  #水平拼接
    none_zero_position = np.array(none_zero_position)  #转换成array格式
    name = 'Data/drug_disease.txt'
    np.savetxt(name, none_zero_position,fmt="%d",delimiter=' ')  #正样本位置，对应的索引值

def Get_embedding_Matrix(gat_feature,drug_dis_matrix):    
    drug_dis_matrix_net = np.mat(constructNet(drug_dis_matrix))  #（2202+1710）×（2202+1710）    
    Net2edgelist(drug_dis_matrix_net)        
    gat_feature = torch.Tensor(gat_feature)  #转换成tensor格式    
    G = np.loadtxt('Data/drug_disease.txt',dtype=int)    
    adj = np.empty([gat_feature.shape[0], gat_feature.shape[0]], dtype=int)
    for i in range(G.shape[0]):
        x = G[i][0]   #正样本中的药物位置
        y = G[i][1]   #正样本中的疾病位置
        adj[x][y] = 1
        adj[y][x] = 1
    adj = torch.Tensor(adj)
    GAL = GAT_layer.GAT(600,68, 100, 0.2, 0.2, 4)  #(n_feat输入特征, n_hid隐藏特征, n_class输出维度, dropout, alpha, n_heads头数)
    matrix = GAL(gat_feature, adj)
    matrix = matrix.detach().numpy()
    return matrix


#读取数据
feature_matrix1 = np.loadtxt('Data/drugfeature1.txt')   #药物特征
feature_matrix2 = np.loadtxt('Data/disfeature1.txt')    #疾病特征
drug_dis_matrix = np.loadtxt('Data/association_matrix.csv',delimiter=',',dtype=float)   #药物-疾病邻接矩阵  

#拼接两个特征矩阵
gat_feature = np.vstack((feature_matrix1,feature_matrix2))   #药物特征矩阵和疾病特征矩阵竖直拼接

#计算嵌入特征（获取拓扑特征）
drug_dis_emb = Get_embedding_Matrix(gat_feature,drug_dis_matrix) 

drug_len = drug_dis_matrix.shape[0]  #miRNA数目        

drug_emb_matrix = np.array(drug_dis_emb[0:drug_len, 0:]) #药物嵌入特征        
dis_emb_matrix = np.array(drug_dis_emb[drug_len::, 0:])  #疾病嵌入特征

#保存数据
np.savetxt('Data/drug_embfeat.txt',drug_emb_matrix,fmt="%d", delimiter=' ')
np.savetxt('Data/dis_embfeat.txt',dis_emb_matrix,fmt="%d", delimiter=' ')

