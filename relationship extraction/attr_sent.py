from torch import nn,optim,utils,autograd
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from gensim.models import Word2Vec
import numpy as np
import jieba

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,auc,confusion_matrix,roc_curve
# class find_relationship(nn.Module):
#     def __init__(self,hidden_size,layers_num,embedding_matrix,ner_model):
#         super(find_relationship, self).__init__()
#         self.embedding=nn.Embedding(num_embeddings=len(embedding_matrix),
#                                     embedding_dim=len(embedding_matrix[0])).\
#             from_pretrained(torch.Tensor(embedding_matrix))
#         self.bilstm=nn.LSTM(bidirectional=True,input_size=self.embedding.embedding_dim,
#                             hidden_size=hidden_size,num_layers=layers_num)
#         self.ner=ner_model
#         # for param in self.ner.parameters():
#         #     param.requires_grad=False
#         self.linear2=nn.Linear(100+1,20)
#         self.linear3=nn.Linear(20,1)
#
#
#     def forward(self,x,attr_index,sent_index):
#         lstm_x=self.ner(x)
#         # attr_x_word,sent_x_word=torch.matmul(attr_index, embed_x),torch.matmul(sent_index,embed_x)
#
#         attr_x,sent_x=torch.matmul(attr_index, lstm_x),torch.matmul(sent_index,lstm_x)
#         cos=torch.cosine_similarity(attr_x,sent_x,dim=2).view(-1,1,1)
#         input_x=torch.mul(attr_x,sent_x)
#         input_x=torch.cat([input_x,cos],dim=2)
#         output_x=self.linear2(input_x)
#         output_x=self.linear3(output_x)
#         return output_x

#NERmodel+MLP
class find_relationship(nn.Module):
    def __init__(self,embedding_matrix,ner_model):
        super(find_relationship, self).__init__()
        self.embedding=nn.Embedding(num_embeddings=len(embedding_matrix),
                                    embedding_dim=len(embedding_matrix[0])).\
            from_pretrained(torch.Tensor(embedding_matrix))
        self.ner=ner_model
        self.mlp1=nn.Linear(in_features=2*self.ner.lstm2.hidden_size+1,out_features=50)
        self.mlp2=nn.Linear(in_features=50,out_features=25)
        self.mlp3=nn.Linear(in_features=25,out_features=1)
        self.dropout=nn.Dropout(0.3)

    def forward(self,x,attr_index,sent_index):
        lstm_x=self.ner(x)
        attr_x,sent_x=torch.matmul(attr_index, lstm_x),torch.matmul(sent_index,lstm_x)
        cos=torch.cosine_similarity(attr_x,sent_x,dim=2).view(-1,1,1)
        input_x=torch.mul(attr_x,sent_x)
        input_x=torch.cat([input_x,cos],dim=2)

        mlp1=self.dropout(nn.functional.relu(self.mlp1(input_x)))
        mlp2=self.dropout(nn.functional.relu(self.mlp2(mlp1)))
        mlp_x=self.mlp3(mlp2)
        return mlp_x





class word_extract(nn.Module):
    def __init__(self,d_model,embedding_matrix):
        super(word_extract, self).__init__()
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
        return second_x



def padding(x, seq_lengths, paddingvalue=0):  # 将sequence填充到最大长度
    for i, per_x in enumerate(x):
        if len(per_x) >= seq_lengths:
            x[i] = per_x[:seq_lengths]
        else:
            x[i] = per_x + [paddingvalue] * (seq_lengths - len(per_x))
    x=torch.LongTensor(x)
    return x

def word2vec_translation(model):  # 负责将数值索引转为要输入的数据
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
    return embeddings_matrix,word2idx,num2idx

def trans_onehot(x,size):#将位置索引转化为one-hot向量，后续可以矩阵相乘直接得出目标向量
    index=torch.LongTensor(x).view(len(x),-1)
    one_hot = torch.zeros(len(index), size).scatter_(1, index, 1)
    one_hot =one_hot.view(one_hot.size(0),-1,one_hot.size(1))
    return one_hot

def train(model,epoch,data,valdata,datalen,valdatalen):
    optimizor=optim.Adam(model.parameters(),lr=10e-3)
    criterion=nn.BCELoss()
    sig=nn.Sigmoid()
    y_true,y_pre,y_pre2=[],[],[]
    for j in range(1,epoch+1):
        accsum,losssum=0,0
        for i,(per_x,per_y,per_attr,per_sent) in enumerate(data):
            optimizor.zero_grad()
            output_y=model(per_x,per_attr,per_sent)
            output_y=sig(output_y)
            y_pred = output_y.ge(0.6).float()
            per_correct = (y_pred.view(-1,1) == per_y.view(-1,1)).sum()
            accsum+=per_correct.item()

            loss=criterion(output_y.view(-1,1),per_y.view(-1,1))
            losssum+=loss.item()
            loss.backward()
            optimizor.step()
        print("The loss of {} iteration is {}".format(j,losssum/datalen))
        print("******The accuracy of {} iteration is {}******".format(j,accsum/datalen))
        if j%5==0:
            valaccsum,valloss=0,0
            for k,(val_x,val_y,val_attr,val_sent) in enumerate(valdata):
                val_output_y = model(val_x, val_attr, val_sent)
                val_output_y = sig(val_output_y)
                val_y_pred=val_output_y.ge(0.6).float()
                correct=(val_y_pred.view(-1,1)==val_y.view(-1,1)).sum()
                valaccsum+=correct.item()
                val_loss=criterion(val_output_y.view(-1,1),val_y.view(-1,1))
                valloss+=val_loss.item()
                if j==epoch:
                    y_true.extend(list(val_y.numpy()))
                    y_pre.extend(list(val_y_pred.view(-1).numpy()))
                    y_pre2.extend(list(val_output_y.view(-1).detach().numpy()))
            print("The val_loss of {} iteration is {}".format(j, valloss / valdatalen))
            print("******The val_accuracy of {} iteration is {}*******".format(j,valaccsum/valdatalen))

    precision=precision_score(y_true=y_true,y_pred=y_pre)
    recall=recall_score(y_true=y_true,y_pred=y_pre)
    f1score=f1_score(y_true=y_true,y_pred=y_pre)
    matrix=confusion_matrix(y_true=y_true,y_pred=y_pre)
    fpr, tpr, thresholds = roc_curve(y_true, y_pre2)
    aucscore=auc(fpr,tpr)
    return model,precision,recall,f1score,matrix

def predict(model,valdata,valdatalen):
    valaccsum,valloss=0,0
    y_true,y_pre,y_pre2=[],[],[]
    criterion=nn.BCELoss()
    sig=nn.Sigmoid()

    for k, (val_x, val_y, val_attr, val_sent) in enumerate(valdata):
        val_output_y = model(val_x, val_attr, val_sent)
        val_output_y = sig(val_output_y)
        val_y_pred = val_output_y.ge(0.61).float()
        correct = (val_y_pred.view(-1, 1) == val_y.view(-1, 1)).sum()
        valaccsum += correct.item()
        val_loss = criterion(val_output_y.view(-1, 1), val_y.view(-1,1))
        valloss += val_loss.item()
        y_true.extend(list(val_y.numpy()))
        y_pre.extend(list(val_y_pred.view(-1).numpy()))

    return y_true,y_pre


def data_process(data):
    attrs=trans_onehot(data['attr'].reset_index(drop=True), size=60)
    sents=trans_onehot(data['sent'].reset_index(drop=True),size=60)
    x=[list(map(int,i.split(' '))) for i in list(data['词向量索引'])]
    x=padding(x,seq_lengths=60)
    y=torch.Tensor(list(data['标签']))
    dataset=torch.utils.data.TensorDataset(x,y,attrs,sents)
    dataset=DataLoader(dataset,batch_size=50,shuffle=True)
    return dataset


if __name__=='__main__':
    file='./尝试数据-1 -删除副词.csv'
    data=pd.read_csv(file,engine='python',encoding='utf-8-sig')
    w2v_model=Word2Vec.load('word2vec_model_100')#读入词向量模型
    embedding_matrix,_,_=word2vec_translation(w2v_model)
    fold=StratifiedKFold(n_splits=10,shuffle=True)
    # fold_data=fold.split(list(data['标签']),list(data['标签']))
    # i=1
    # pl,rl,f1l=[],[],[]
    # for fit,val in fold_data:
    #     train_data,test_data=data.iloc[fit],data.iloc[val]
    #     train_dataset=data_process(train_data)
    #     test_dataset=data_process(test_data)
    #     ner_model=word_extract(d_model=200,embedding_matrix=embedding_matrix)
    #     ner_model.load_state_dict(torch.load('extract_model_bilstm.pkl'))
    #     classifier=find_relationship(embedding_matrix=embedding_matrix,ner_model=ner_model)
    #     classifier_model,p,r,f1,matrix=train(model=classifier,epoch=15,data=train_dataset,
    #                            valdata=test_dataset,datalen=len(train_data),valdatalen=len(test_data))
    #     pl.append(p)
    #     rl.append(r)
    #     f1l.append(f1)
    #     print("{} Fold".format(i))
    #     print("Precision:{},Recall:{},f1score:{},AUC:{}".format(p,r,f1,auc))
    #     print("confusion_matrix:{}".format(matrix))
    #     torch.save(classifier.state_dict(),'classifier_{}.pkl'.format(i))
    #     i+=1
    # res=pd.DataFrame()
    # res['fold']=list(range(1,11))
    # res['precision']=pl
    # res['recall']=rl
    # res['f1score']=f1l
    # res.to_csv('./res.csv',index=False)
    # print('s')


    '''模型验证集预测'''
    pl,rl,f1l=[],[],[]
    ner_model = word_extract(d_model=200, embedding_matrix=embedding_matrix)
    classifier = find_relationship(embedding_matrix=embedding_matrix, ner_model=ner_model)
    for i in range(1,11):
        y_true,y_pred=[],[]
        classifier.load_state_dict(torch.load('classifier_{}.pkl'.format(i)))
        fold_data = fold.split(list(data['标签']), list(data['标签']))
        for fit, val in fold_data:
            train_data, test_data = data.iloc[fit], data.iloc[val]
            test_dataset = data_process(test_data)
            y_t,y_p=predict(model=classifier,valdata=test_dataset,valdatalen=len(test_data))
            y_true.extend(y_t)
            y_pred.extend(y_p)

        p=precision_score(y_true=y_true,y_pred=y_pred)
        r=recall_score(y_true=y_true,y_pred=y_pred)
        f1=f1_score(y_true=y_true,y_pred=y_pred)
        matrix=confusion_matrix(y_true=y_true,y_pred=y_pred)
        print("Precision:{},Recall:{},f1score:{},".format(p,r,f1))
        print("confusion_matrix:{}".format(matrix))
        pl.append(p)
        rl.append(r)
        f1l.append(f1)
    res=pd.DataFrame()
    res['fold']=list(range(1,11))
    res['precision'],res['recall'],res['f1score']=pl,rl,f1l
    res.to_csv('./val_res.csv',header=True,index=False)
    print('s')