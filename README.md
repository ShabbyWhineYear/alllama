# alllama
尝试写个能在小显存上跑的支持中文的模型当作langchain的llm引擎

结构参考了llama2

没有使用常规的Embedding，随便选了一个字体文件，把字体文件内每个字符转换为01点阵，把点阵拉直成一维向量作为该字符的词向量。vocab就是字体文件能显示的所有字符。相应的模型输出也是sigmoid后的01向量，与存在FAISS中的所有字体文件字符向量比较，把排第一的字符作为预测值

预训练过程是用0-3号字符预测1-4号字符、用0-4号字符预测1-5号字符、用0-5号字符预测1-6号字符，以此类推

训练语料来自[awesome-chinese-nlp](https://github.com/crownpku/awesome-chinese-nlp)项目，全过程无监督训练
