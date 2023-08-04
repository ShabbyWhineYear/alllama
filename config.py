class Config():
    def __init__(self):
        self.data_path = "./data/train.txt" # 训练集要求是文字格式，每行2个句子，用\t分隔，长度超过max_seq_len的部分会被截断
        self.output_path = "./output/model.bin"
        self.word_dim = 400 # 常用的是每个分词用一个数字id表示。我的每个分词的id是400维向量，这个参数是我特有的
        self.embedding_dim = 256 # 嵌入后词向量维度。原始llama2是4096
        self.hidden_dim_attention = 1152 # 隐藏层维度。原始llama2是4096，和embedding_dim相同所以只有一个配置参数。我分开是为了后续做albert的因式分解
        self.max_seq_len = 512 # 最大序列长度。原始llama2是2048
        self.n_layers = 18 # 解码器层数。原始llama2是32
        self.n_head_groups = 4 # 注意力组数。原始llama2是16
        self.n_heads_per_group = 4 # 每组注意力头数，同组中共享k和v，每个头的q不同。原始llama2是2
        self.hidden_dim_feedforward = 2048 # 前馈隐藏层维度。原始llama2是256*((4096+256-1)//256)=16384
        self.dropout = 0.0 # 原始llama2是0.0
        self.eps = 1e-10 # 原始llama2是1e-5
        self.device = 'cuda'
        self.batch_size = 1
        self.weight_decay = 0.1 # 权重衰减系数
        self.learning_rate = 1e-4 # 学习率
        self.beta1 = 0.9 # beta1和beta2是adamw优化器的衰减率参数
        self.beta2 = 0.95
        self.epochs = 1000
        try:
            assert self.hidden_dim_attention % (self.n_head_groups*self.n_heads_per_group) == 0
        except Exception as e:
            print("config.py中的hidden_dim_attention必须是n_head_groups*n_heads_per_group的整数倍")
            exit(0)

if __name__ == '__main__':
    conf = Config()
