import torch
import pandas as pd
import numpy as np
# from tensorflow import keras
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import gensim
import itertools
from torchcrf import CRF
import copy
word2vec=gensim.models.Word2Vec.load('./word2vec_model_100')
from sklearn.metrics import f1_score,precision_score,recall_score\
    ,auc,multilabel_confusion_matrix,confusion_matrix
from sklearn.model_selection import KFold,StratifiedKFold,ShuffleSplit
class word_extract_crf(nn.Module):
    def __init__(self,d_model,embedding_matrix):
        super(word_extract_crf, self).__init__()
        self.d_model=d_model
        self.embedding=nn.Embedding(num_embeddings=len(embedding_matrix),embedding_dim=200)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad=False
        self.lstm1=nn.LSTM(input_size=200,hidden_size=50,bidirectional=True)
        self.lstm2=nn.LSTM(input_size=2*self.lstm1.hidden_size,hidden_size=50,bidirectional=True)
        self.linear=nn.Linear(2*self.lstm2.hidden_size,4)
        self.crf=CRF(num_tags=4,batch_first=True)
    def forward(self,x,y,mask):
        w_x=self.embedding(x)
        first_x,(first_h_x,first_c_x)=self.lstm1(w_x)
        second_x,(second_h_x,second_c_x)=self.lstm2(first_x)
        output_x=self.linear(second_x)
        # softmax_x=F.softmax(output_x)
        loss=self.crf(output_x,y,mask=mask)
        return -loss,output_x
class word_extract_bilstm(nn.Module):
    def __init__(self,d_model,embedding_matrix):
        super(word_extract_bilstm, self).__init__()
        self.d_model=d_model
        self.embedding=nn.Embedding(num_embeddings=len(embedding_matrix),embedding_dim=200)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad=False
        self.lstm1=nn.LSTM(input_size=200,hidden_size=50,bidirectional=True)
        self.lstm2=nn.LSTM(input_size=2*self.lstm1.hidden_size,hidden_size=50,bidirectional=True)
        self.linear=nn.Linear(2*self.lstm2.hidden_size,4)

    def forward(self,x):
        w_x=self.embedding(x)
        first_x,(first_h_x,first_c_x)=self.lstm1(w_x)
        second_x,(second_h_x,second_c_x)=self.lstm2(first_x)
        output_x=self.linear(second_x)
        return output_x


def establish_word2vec_matrix(model):  #负责将数值索引转为要输入的数据
    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    num2idx = {0: "_PAD"}
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        num2idx[i + 1] = word
        embeddings_matrix[i + 1] = vocab_list[i][1]
    embeddings_matrix = torch.Tensor(embeddings_matrix)
    return embeddings_matrix, word2idx, num2idx

def trans_num(word2idx,text):
    text_list=[]
    for i in text:
        s=i.rstrip().replace('\r','').replace('\n','').split(' ')
        numtext=[word2idx[j] if j in word2idx.keys() else word2idx['_PAD'] for j in s ]
        text_list.append(numtext)
    return text_list

def padding(x, seq_lengths,paddingvalue=0):  # 将sequence填充到最大长度
    mask_x=[]
    for i,per_x in enumerate(x):
        if len(per_x)>=seq_lengths:
            x[i]=per_x[:seq_lengths]
            mask_x.append([1]*seq_lengths)
        else:
            x[i]=per_x+[paddingvalue]*(seq_lengths-len(per_x))
            mask_x.append([1]*len(per_x)+[0]*(seq_lengths-len(per_x)))

    # x=torch.LongTensor(x)
    return x,mask_x

def train_bilstm(model,epoch,learning_rate,batch_size,x, y, val_x, val_y):
    optimizor = optim.Adam(model.parameters(), lr=learning_rate)
    data = TensorDataset(x, y)
    data = DataLoader(data, batch_size=batch_size)
    for i in tqdm(range(epoch)):
        for j, (per_x, per_y) in enumerate(data):
            output_y = model(per_x)
            loss = F.cross_entropy(output_y.view(-1,output_y.size(2)), per_y.view(-1))
            optimizor.zero_grad()
            loss.backward()
            optimizor.step()
            arg_y=output_y.argmax(dim=2)
            fit_correct=(arg_y==per_y).sum()
            fit_acc=fit_correct.item()/(per_y.size(0)*per_y.size(1))
            # print('##################################')
            # print('第{}次迭代第{}批次的训练误差为{}'.format(i + 1, j + 1, loss), end=' ')
            # print('第{}次迭代第{}批次的训练准确度为{}'.format(i + 1, j + 1, fit_acc))
            val_output_y = model(val_x)
            val_loss = F.cross_entropy(val_output_y.view(-1,val_output_y.size(2)), val_y.view(-1))
            arg_val_y=val_output_y.argmax(dim=2)
            val_correct=(arg_val_y==val_y).sum()
            val_acc=val_correct.item()/(val_y.size(0)*val_y.size(1))
            print('第{}次迭代第{}批次的预测误差为{}'.format(i + 1, j + 1, val_loss), end=' ')
            print('第{}次迭代第{}批次的预测准确度为{}'.format(i + 1, j + 1, val_acc))

    torch.save(model,'./extract_model_{}.pkl'.format('Bilstm'))
    #进行评估指标的计算
    y_true,y_pred=val_y.view(-1),arg_val_y.view(-1)
    precision=precision_score(y_true,y_pred,average='macro')
    f1score=f1_score(y_true,y_pred,average='macro')
    recall=recall_score(y_true,y_pred,average='macro')
    print('precision:{},recall:{},f1score:{}'.format(precision,recall,f1score))
    return precision,recall,f1score,y_true.numpy(),y_pred.numpy()

def train_crf(model,epoch,learning_rate,batch_size,x, y, val_x, val_y,mask_fit,mask_val,model_name='Bi-lstm+CRF'):
    optimizor = optim.Adam(model.parameters(), lr=learning_rate)
    # for parameter in model.parameters():
    #     print(parameter)
    data = TensorDataset(x, y,mask_fit)
    data = DataLoader(data, batch_size=batch_size)
    for i in tqdm(range(epoch)):
        for j, (per_x, per_y,per_mask) in enumerate(data):
            loss,output_y = model(per_x,per_y,per_mask)
            model.zero_grad()
            # loss= model.crf(output_y,per_y,mask=per_mask)
            # loss = F.cross_entropy(output_y.view(-1,output_y.size(2)), per_y.view(-1))
            loss.backward()
            optimizor.step()
            tensory=model.crf.decode(output_y, mask=per_mask)
            # arg_y=torch.tensor(tensory)
            # fit_correct=(arg_y==per_y).sum()
            # fit_acc=fit_correct.item()/(per_y.size(0)*per_y.size(1))
            # print('##################################')
            # print('第{}次迭代第{}批次的训练误差为{}'.format(i + 1, j + 1, loss), end=' ')
            # print('第{}次迭代第{}批次的训练准确度为{}'.format(i + 1, j + 1, fit_acc))
            val_loss,val_output_y = model(val_x,val_y,mask=mask_val)
            # val_loss = F.cross_entropy(val_output_y.view(-1,val_output_y.size(2)), val_y.view(-1))
            val_output_y_crf=model.crf.decode(val_output_y)
            arg_val_y=torch.tensor(val_output_y_crf)
            val_correct=(arg_val_y==val_y).sum()
            val_acc=val_correct.item()/(val_y.size(0)*val_y.size(1))
            print('第{}次迭代第{}批次的预测误差为{}'.format(i + 1, j + 1, val_loss), end=' ')
            print('第{}次迭代第{}批次的预测准确度为{}'.format(i + 1, j + 1, val_acc))

    torch.save(model,'./extract_model_{}.pkl'.format(model_name))
    #进行评估指标的计算
    y_true,y_pred=val_y.view(-1),arg_val_y.view(-1)
    precision=precision_score(y_true,y_pred,average='macro')
    f1score=f1_score(y_true,y_pred,average='macro')
    recall=recall_score(y_true,y_pred,average='macro')
    print('precision:{},recall:{},f1score:{}'.format(precision,recall,f1score))

    return precision,recall,f1score,y_true.numpy(),y_pred.numpy()

def function(train_x,train_y,mask_train_x,res):
    p_list1,r_list1,f_list1,y_t1,y_p1=[],[],[],[],[]
    p_list2, r_list2, f_list2, y_t2, y_p2 = [], [], [], [], []
    for fit,val in res:
        fit_x,fit_y=list(train_x.iloc[fit].values),list(train_y.iloc[fit].values)
        val_x,val_y=list(train_x.iloc[val].values),list(train_y.iloc[val].values)

        fit_x,fit_y=torch.LongTensor(fit_x),torch.LongTensor(fit_y)
        val_x,val_y=torch.LongTensor(val_x),torch.LongTensor(val_y)
        w_extract_1=word_extract_bilstm(d_model=200,embedding_matrix=embedding_matrix)
        p1,r1,f1,y_true1,y_pred1=train_bilstm(model=w_extract_1,epoch=8,learning_rate=0.001,batch_size=50,
          x=fit_x,y=fit_y,val_x=val_x,val_y=val_y)
        p_list1.append(p1)
        r_list1.append(r1)
        f_list1.append(f1)
        y_t1.extend(y_true1)
        y_p1.extend(y_pred1)
        print('进行bilstm+crf训练')
        mask_fit_x = list(mask_train_x.iloc[fit].values)
        mask_val_x = list(mask_train_x.iloc[val].values)
        mask_val_x, mask_fit_x = torch.tensor(mask_val_x, dtype=torch.uint8), torch.tensor(mask_fit_x,
                                                                                           dtype=torch.uint8)
        w_extract=word_extract_crf(d_model=200,embedding_matrix=embedding_matrix)
        p2,r2,f2,y_true2,y_pred2=train_crf(model=w_extract,epoch=8,learning_rate=0.001,batch_size=50,
          x=fit_x,y=fit_y,val_x=val_x,val_y=val_y,mask_fit=mask_fit_x,mask_val=mask_val_x)
        p_list2.append(p2)
        r_list2.append(r2)
        f_list2.append(f2)
        y_t2.extend(y_true2)
        y_p2.extend(y_pred2)
    matrix1 = confusion_matrix(y_t1, y_p1)
    matrix2 = confusion_matrix(y_t2, y_p2)
    print(matrix1)
    print(matrix2)
    result=pd.DataFrame()
    result['precision']=p_list2
    result['recall']=r_list2
    result['f1score']=f_list2
    result.to_csv('./res_crf.csv',index=False)

    result=pd.DataFrame()
    result['precision']=p_list1
    result['recall']=r_list1
    result['f1score']=f_list1
    result.to_csv('./res_bilstm.csv',index=False)
    return matrix1,matrix2
if __name__=='__main__':
    #数据准备过程
    #生成词向量矩阵
    embedding_matrix,word2idx,num2idx=establish_word2vec_matrix(word2vec)
    #训练数据
    train_data=pd.read_csv('./数据.csv',encoding='utf-8-sig',engine='python')
    train_data=train_data.sample(frac=1)
    train_x=list(train_data['文本'])
    train_x=trans_num(word2idx,train_x)
    train_x,mask_train_x=padding(train_x,seq_lengths=60,paddingvalue=0)
    train_y=list(train_data['BIO数值'])

    y_text=[]
    for i in train_y:
        s=i.rstrip().split(' ')
        numtext=[int(j) for j in s]
        y_text.append(numtext)
    train_y=y_text
    train_y,_=padding(train_y,seq_lengths=60,paddingvalue=3)
    fold=StratifiedKFold(n_splits=10,shuffle=True)
    check=[1]*len(train_y)
    res=fold.split(X=train_x,y=check)
    mask_train_x=pd.DataFrame(data=mask_train_x)
    train_x,train_y=pd.DataFrame(data=train_x),pd.DataFrame(data=train_y)
    matrix1,matrix2=function(train_x,train_y,mask_train_x,res)
    print('s')


