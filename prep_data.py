import json
from tqdm import tqdm
import re
import char_utils
import pickle

path1='./data/web_text_zh_testa.json'
path2='./data/web_text_zh_train.json'
path3='./data/web_text_zh_valid.json'

def clean_data(text):
    text = text.replace('\n', '')
    text = re.sub(r'。+', '。', text)
    text = re.sub(r'--+[^-]*?--+', '', text)
    text = re.sub(r'<.*?b.*?r.*?>', '', text)
    return text

def json2txt():
    with open('./data/train.txt','w',encoding='utf-8') as fout:
        with open(path1,'r',encoding='utf-8') as fin1:
            lines=tqdm(fin1.readlines())
            for line in lines:
                line=json.loads(line)
                x = clean_data(line['title']+line['desc'])
                y = clean_data(line['content'])
                fout.write(x+'\t'+y+'\n')
        with open(path2,'r',encoding='utf-8') as fin2:
            lines=tqdm(fin2.readlines())
            for line in lines:
                line=json.loads(line)
                x = clean_data(line['title']+line['desc'])
                y = clean_data(line['content'])
                fout.write(x+'\t'+y+'\n')
        with open(path3,'r',encoding='utf-8') as fin3:
            lines=tqdm(fin3.readlines())
            for line in lines:
                line=json.loads(line)
                x = clean_data(line['title']+line['desc'])
                y = clean_data(line['content'])
                fout.write(x+'\t'+y+'\n')

def txt2bin():
    with open('./data/train.txt', 'r', encoding='utf-8') as fin1:
        x = []
        y = []
        lines = tqdm(fin1.readlines())
        train_cnt = 0
        for cnt,line in enumerate(lines):
            tmp_x = []
            tmp_y = []
            question, answer = line.split('\t')
            for i in range(len(question)):
                tmp_x.append(char_utils.char_to_vector(question[i]))
            for i in range(len(answer)):
                tmp_y.append(char_utils.char_to_vector(answer[i]))
            tmp_y.append(char_utils.char_to_vector('EOS'))
            x.append(tmp_x)
            y.append(tmp_y)
            lines.set_description('Processing %s' % (len(question)+len(answer)))
            if cnt % 1000 == 999:
                pickle.dump((x, y), open('./data/train_%s.bin' % train_cnt, 'wb'))
                train_cnt += 1
                x = []
                y = []



if __name__ == '__main__':
    # json2txt()
    txt2bin() # txt2bin()跑的太慢了，跑完训练集需要一百多个小时，边训练边转吧


