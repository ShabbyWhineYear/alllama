from model import MyModel
import numpy as np
import torch
import char_utils
import random
from config import Config
conf = Config()
from torch.utils.data import Dataset
import pickle

index, char_library = pickle.load(open('faiss_char.pkl', 'rb'))

class MyDataset(Dataset):
    def __init__(self,path='./data/train.txt'):
        self.x = []
        self.y = []
        with open(path,'r',encoding='utf-8') as fin1:
            for lines in fin1.readlines():
                if len(lines.split('\t'))!=2:
                    continue
                x,y=lines.split('\t')
                self.x.append(x)
                self.y.append(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


class MyDataLoader():
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.length = len(self.dataset)
        self.batch_size = conf.batch_size
        self.shuffle = shuffle
        self.idx = 0
        self.indices = list(range(self.length))
        if self.shuffle:
            random.shuffle(self.indices)
    def sentence2vector(self,sentence,pad=0):
        vector = []
        for i in range(len(sentence)):
            vector.append(char_utils.char_to_vector(sentence[i]))
        vector.append(char_utils.char_to_vector('EOS'))
        for i in range(pad):
            vector.append(np.array([0 for _ in range(conf.word_dim)]))
        return vector
    def __len__(self):
        # 总批次数
        return (self.length + self.batch_size - 1) // self.batch_size
    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self
    def __next__(self):
        if self.idx >= self.length:
            raise StopIteration
        else:
            batch_indices = self.indices[self.idx:self.idx+self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            self.idx += self.batch_size
            samples = [self.sentence2vector(data[0][:conf.max_seq_len-2])+
                 self.sentence2vector(data[1][:conf.max_seq_len-len(data[0])-2],pad=conf.max_seq_len-len(data[0])-len(data[1])-2)
                 for data in batch]
            mask = [[1 for _ in range(min(conf.max_seq_len,len(data[0])+1+len(data[1])+1))]+[0 for _ in range(conf.max_seq_len-len(data[0])-len(data[1])-2)] for data in batch]
            return samples, mask


def train(path):
    model = MyModel().to(conf.device)
    optimizer = model.get_optimizer()
    dataset = MyDataset(path)
    dataloader = MyDataLoader(dataset,shuffle=True)
    for epoch in range(conf.epochs):
        loss_total = 0.0
        sample_cnt = 0
        for samples,mask in dataloader:
            i = 3
            loss_tmp = 0.0
            x = torch.tensor(samples,dtype=torch.float32,device=conf.device)
            y_true = torch.cat([x[:,1:,:],(torch.tensor(char_utils.char_to_vector('EOS'),device=conf.device).unsqueeze(0).unsqueeze(0)).repeat(conf.batch_size,1,1)],dim=1)
            while i < conf.max_seq_len-1 and sum([mask[j][i] for j in range(len(mask))]) > 0:
                y_mask = torch.tensor([[1 for _ in range(i)]+[0 for _ in range(conf.max_seq_len-i)] for _ in range(len(samples))],dtype=torch.float32,device=conf.device)
                y_mask = y_mask * torch.tensor(mask,dtype=torch.float32,device=conf.device)
                loss,out_vec = model(x,(y_true,y_mask.unsqueeze(2)))
                loss_tmp += loss.item()*conf.batch_size
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
                print('\r第%s轮，第%s批样本，第%s个字符，当前批次损失：%.4f。真实值：%s，预测值：%s'%(epoch+1,sample_cnt+1,i,loss_tmp/(i-2),char_utils.find_closest_char(y_true.cpu().detach().numpy()[0,i,:],index,char_library),char_utils.find_closest_char(out_vec.cpu().detach().numpy()[0,i,:],index,char_library)),end='')
                i += 1
            print("")
            loss_total += loss_tmp
            sample_cnt += conf.batch_size
            if sample_cnt % 1000 == 900:
                torch.save(model.state_dict(), conf.output_path)
                print('\n模型已保存，当前总损失：%.4f'%(loss_total/sample_cnt))









if __name__ == '__main__':
    # dataset = []
    # dataset.append(('你好'*400,'你好啊'))
    # dataset.append(('我很好','我很好啊'))
    # dataloader = MyDataLoader(dataset)
    # for samples,mask in dataloader:
    #     print(samples)
    #     print(mask)
    #     break
    train(conf.data_path)
