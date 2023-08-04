# encoding: utf-8
# 模型结构是根据这个来的：https://zhuanlan.zhihu.com/p/636784644
import math

import torch.nn.functional as F

from config import Config

conf = Config()
import torch

print('Using device:', conf.device)


class RMSNorm(torch.nn.Module):
    """
    与常见的标准化层LayerNorm的区别是把分母上的方差换成了平方的均值；分子上不再减去一个均值；整体不再加偏置
    """

    def __init__(self, dim=conf.hidden_dim_attention, eps=conf.eps):
        super().__init__()
        self.eps = eps  # eps用来防止分母为0
        self.weight = torch.nn.Parameter(torch.ones(dim,device=conf.device))  # 加权标准化里的权重，初始值为1

    def _norm(self, x):
        """
        如果是常规的LayerNorm，完整的表达式是：
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) * self.weight + self.bias
        """
        return x / (torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)+self.eps) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def get_position_matrix(dim=conf.hidden_dim_attention, max_seq_len=conf.max_seq_len, theta_base=10000.0):
    """
    生成位置编码矩阵
    :param theta_base: theta_base越大则位置编码的频率变化越缓慢，模型更容易学习到长距离的依赖关系；theta越小则位置编码的频率变化越激烈，模型更容易学习到短距离的依赖关系
    """
    theta = 1.0 / (theta_base ** (torch.arange(0, dim, 2,device=conf.device) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32,device=conf.device)
    freqs = torch.einsum("i,j->ij", t, theta)
    freqs = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
    _sin = torch.sin(freqs)
    _cos = torch.cos(freqs)
    return (_cos, _sin)


def embed_position(x, position_matrix):
    """
    把位置编码嵌入到输入张量x中
    计算方法是用x乘以cos位置编码矩阵，然后把x从中间切开，下半部分取反后拼到上半部分的上面，再乘以sin位置编码矩阵。把两个乘积加起来得到嵌入了位置编码的张量
    """
    dim = x.shape[-1]
    _x = torch.cat((-x[:, :, dim // 2:], x[:, :, :dim // 2]), dim=-1)
    return (_x * position_matrix[0][:, :, :dim] + _x * position_matrix[1][:, :, :dim]).type_as(x)


# 前馈全连接层（结构参考顶部知乎链接）
class FeedForward(torch.nn.Module):
    def __init__(self, dim=conf.hidden_dim_attention, dim_feedforward=conf.hidden_dim_feedforward,
                 dropout=conf.dropout):
        super().__init__()
        self.gate = torch.nn.Linear(dim, dim_feedforward, bias=False,device=conf.device)
        self.down = torch.nn.Linear(dim_feedforward, dim, bias=False,device=conf.device)
        self.up = torch.nn.Linear(dim, dim_feedforward, bias=False,device=conf.device)
        self.dropout = torch.nn.Dropout(dropout).to(conf.device)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# 多头注意力层
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head_groups=conf.n_head_groups, attention_dim=conf.hidden_dim_attention, dropout=conf.dropout):
        super(MultiHeadAttention, self).__init__()
        self.wv = torch.nn.Linear(attention_dim, attention_dim // n_head_groups, bias=False,device=conf.device)  # v和k会在组内共享，所以要除以组数
        self.wk = torch.nn.Linear(attention_dim, attention_dim // n_head_groups, bias=False,device=conf.device)
        self.wq = torch.nn.Linear(attention_dim, attention_dim, bias=False,device=conf.device)
        self.wo = torch.nn.Linear(attention_dim, attention_dim, bias=False,device=conf.device)
        self.dropout = torch.nn.Dropout(dropout)
        mask = torch.full((1, 1, conf.max_seq_len, conf.max_seq_len), 1e-10,requires_grad=False,device=conf.device)  # 生成一个全负无穷的方阵
        self.mask = torch.triu(mask, diagonal=1)  # 把下三角部分置为0，diagnoal=1表示包括主对角线

    def forward(self, x, position_matrix):
        # 准备好马上被分头的qkv
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        # 给q和k加上位置编码
        xq = embed_position(xq, position_matrix)
        xk = embed_position(xk, position_matrix)
        # k和v会在组内共享，所以先拆成组数，再每组各自复制成每组头数。转置把头的维度前置，方便每个头内算注意力
        xk = (xk[:, :, None, :]
              .expand(conf.batch_size, conf.max_seq_len, conf.n_head_groups, -1)
              .reshape(conf.batch_size,conf.max_seq_len,conf.n_head_groups * conf.n_heads_per_group,-1)
              .transpose(1, 2))
        xv = (xv[:, :, None, :]
              .expand(conf.batch_size, conf.max_seq_len, conf.n_head_groups, -1)
              .reshape(conf.batch_size, conf.max_seq_len, conf.n_head_groups * conf.n_heads_per_group, -1)
              .transpose(1, 2))
        # q直接按照总头数拆分
        xq = (xq
              .reshape(conf.batch_size, conf.max_seq_len, conf.n_head_groups * conf.n_heads_per_group, -1)
              .transpose(1, 2))
        # 每个头内计算注意力权重
        score = torch.matmul(xq, xk.transpose(2, 3)) / (
                math.sqrt(conf.hidden_dim_attention // conf.n_head_groups) + conf.eps)
        # 通过加上mask让不关注的地方变成负无穷
        score += self.mask.repeat(conf.batch_size, conf.n_head_groups * conf.n_heads_per_group, 1, 1)
        # 使用改良后的softmax_one（见 https://www.jiqizhixin.com/articles/2023-07-25）
        scores = (torch.exp(score.float()) / (torch.sum(torch.exp(score.float()), dim=-1).unsqueeze(-1) + 1)).type_as(
            xq)
        # 加上dropout
        scores = self.dropout(scores)
        # 得到加权求和后的注意力
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).reshape(conf.batch_size, conf.max_seq_len, conf.hidden_dim_attention)
        output = self.wo(output)
        output = self.dropout(output)
        return output

# 结构参考页首的知乎文章
class TransformerBlock(torch.nn.Module):
    def __init__(self,idx):
        super(TransformerBlock,self).__init__()
        self.idx = idx
        self.norm_attention = RMSNorm()
        self.attn = MultiHeadAttention()
        self.norm_feedforward = RMSNorm()
        self.feedforward = FeedForward()
    def forward(self,x,position_matrix):
        x = x + self.attn(self.norm_attention(x),position_matrix)
        x = x + self.feedforward(self.norm_feedforward(x))
        return x

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.position_matrix = get_position_matrix()
        # 如果用的常规tokenizer，自己改成常规torch.nn.Embedding()
        self.embedding = torch.nn.Linear(conf.word_dim,conf.embedding_dim, bias=False)
        self.prep = torch.nn.Linear(conf.embedding_dim,conf.hidden_dim_attention, bias=False)
        self.transformers = []
        for i in range(conf.n_layers):
            self.transformers.append(TransformerBlock(i))
        self.norm = RMSNorm()
        # 如果用的常规tokenizer，这里直接映射到vocab_size，不用过o2线性层
        self.o1 = torch.nn.Linear(conf.hidden_dim_attention,conf.embedding_dim,bias=False)
        # 我在后面会用0.5阈值做二分类，所以这个线性层加了偏置。
        self.o2 = torch.nn.Linear(conf.embedding_dim,conf.word_dim)
        # 你们用不到这个二分类
        self.sigmoid = torch.nn.Sigmoid()
        self.bceloss = torch.nn.BCELoss()
    def forward(self,x,y=None):
        x = self.embedding(x)
        x = self.prep(x)
        for t in self.transformers:
            x = t(x,self.position_matrix)
        x = self.norm(x)
        x = self.o1(x)
        # 下面这部分是我的模型特有的输出部分
        x = self.o2(F.silu(x))
        x = self.sigmoid(x)
        if y is not None:
            y_true = y[0]
            y_mask = y[1]
            loss = self.bceloss(x * y_mask, y_true * y_mask)
            return loss,x
        else:
            return x

    def get_optimizer(self):
        # 初始化参数列表
        params = []
        # 参与梯度下降的参数字典
        param_dict = {param_name:param for param_name,param in self.named_parameters() if param.requires_grad}
        # 维度大于2的参数需要加上正则化
        params_decay = [param for param in param_dict.values() if param.dim() >= 2]
        params_nodecay = [param for param in param_dict.values() if param.dim() < 2]
        params.append({'params':params_decay,'weight_decay':conf.weight_decay})
        params.append({'params':params_nodecay,'weight_decay':0.0})
        # 优化器
        optimizer = torch.optim.AdamW(params,lr=conf.learning_rate,betas=(conf.beta1,conf.beta2))
        return optimizer



if __name__ == '__main__':
    t1 = torch.randn(conf.batch_size, conf.max_seq_len, 400)
    y_true = torch.randn(conf.batch_size, conf.max_seq_len, 400)
    y_mask = torch.ones_like(y_true)
    # position_matrix = get_position_matrix()
    # t2 = embed_position(t1, position_matrix)
    # print(t2.shape)
    # attn = MultiHeadAttention()
    # t3 = attn(t1, get_position_matrix())
    model = MyModel()
    loss = model(t1,(y_true, y_mask))
    print(loss)
    # y_pred = model(t1)
    # print(y_pred.shape)
