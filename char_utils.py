import faiss
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pickle

def char_to_matrix(char, size=20, font_path='/home/liji/source/font/msyhl.ttc'):
    """
    将字符转换为点阵
    :param char: 字符
    :param size: 点阵规模
    :param font_path: 字体文件路径
    :return: np.array格式的点阵
    """
    # 创建一个空白的图像
    image = Image.new('L', (size, size), color=1)
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 加载字体
    font = ImageFont.truetype(font_path, size)
    # 在图像上绘制字符
    draw.text((-size * 0, -size * 0.25), char, fill=0, font=font)
    # 将图像转换为点阵
    matrix = np.array(image)
    return matrix

def char_to_vector(char, size=20, font_path='/home/liji/source/font/msyhl.ttc'):
    """
    将字符转换为向量
    :param char: 字符
    :param size: 点阵规模
    :param font_path: 字体文件路径
    :return: np.array格式的向量
    """
    matrix = char_to_matrix(char, size, font_path)
    vector = np.array(matrix).flatten()
    return vector

from fontTools.ttLib import TTFont
def get_supported_chars(font_path, font_index=0):
    """
    获取字体文件中支持的字符
    :param font_path: 字体文件路径
    :param font_index: ttc字体文件中会有多个字体，font_index是字体索引
    :return: 支持的字符列表
    """
    font = TTFont(font_path, fontNumber=font_index)
    chars = []
    for table in font['cmap'].tables:
        for k,v in table.cmap.items():
            chars.append(chr(k))
        chars.append('CLS') # 实际只能装下前2个字母，CLS和EOS的S可有可无
        chars.append('EOS')
    return chars



def create_faiss_index(size=20, font_path='/home/liji/source/font/msyhl.ttc'):
    """
    创建faiss索引
    :param size: 点阵规模
    :param font_path: 字体文件路径
    :return: faiss索引和字符库
    """
    supported_chars = get_supported_chars(font_path)

    char_vectors = []
    char_library = []

    for char in supported_chars:
        try:
            matrix = char_to_matrix(char, size=size, font_path=font_path)
            vector = np.array(matrix).flatten()
            char_vectors.append(vector)
            char_library.append(char)
        except Exception as e:
            print(f"Error for character {char}: {str(e)}")

    char_vectors = np.vstack(char_vectors).astype('float32')
    index = faiss.IndexFlatL2(char_vectors.shape[1])
    index.add(char_vectors)

    return index, char_library

def find_closest_char(vector, index, char_library):
    """
    通过faiss索引找到与vector最接近的字符
    :param vector: 拉直后的点阵格式的字符
    :param index: faiss索引
    :param char_library: 字符库
    :return: 最接近的字符
    """
    vector = np.array(vector).astype('float32').reshape(1, -1)
    D, I = index.search(vector, 1)
    return char_library[I[0][0]]


import matplotlib.pyplot as plt
def draw_bitmap(bitmap):
    """
    把点阵画出来
    :param bitmap: np.array格式的点阵
    """
    # 创建一个新的图形
    plt.figure()
    # 使用imshow来显示图片
    plt.imshow(bitmap, cmap='gray')
    # 隐藏坐标轴
    plt.axis('off')
    # 显示图形
    plt.show()


if __name__ == '__main__':
    # index,char_library = create_faiss_index()
    # faiss_char = (index, char_library)
    # pickle.dump(faiss_char, open('faiss_char.pkl', 'wb'))
    faiss_char = pickle.load(open('faiss_char.pkl', 'rb'))
    index, char_library = faiss_char
    char_matrix = char_to_matrix('BAK')
    draw_bitmap(char_matrix)
    # 随机从矩阵里抽100次像素点，把该点的0变成1或者1变成0
    for i in range(100):
        x = np.random.randint(0, 20)
        y = np.random.randint(0, 20)
        char_matrix[x][y] = 1 - char_matrix[x][y]
    draw_bitmap(char_matrix)
    char_vector = np.array(char_matrix).flatten()
    closest_char = find_closest_char(char_vector, index, char_library)
    print(closest_char)
