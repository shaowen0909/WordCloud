# preprocessing text file
from zhon.hanzi import punctuation
import jieba
import re
import string

# word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# shape
from PIL import Image
import numpy as np


def preprocessing_file(source_file_path):
    # Read the whole text.
    with open(source_file_path, 'r') as file:
        text = file.read()

    # Remove Chinese punctuation.
    text = re.sub(r"[%s]+" % punctuation, ' ', text)
    # Remove English punctuation.
    chars = re.escape(string.punctuation)
    text = re.sub(r'[' + chars + ']', ' ', text)
    # Remove extra whitespaces.
    text = ' '.join(text.split())

    # Processing Chinese sentences.
    words = jieba.lcut(text)
    result = ''
    for i in words:
        result += str(i)+' '

    # print(result)
    return result


def img_generate(source_file_path, func):
    words = func(source_file_path)

    # Read the mask image.
    mask = np.array(Image.open("../alice_mask.png"))
    stop_words = set()
    with open('../stop_words.txt', 'r') as file:
        for word in file:
            stop_words.add(word.strip('\n'))
    print(stop_words)

    wc = WordCloud(background_color="white", max_words=2000, mask=mask,
                   contour_width=3, scale=2, contour_color='steelblue',
                   font_path="../fonts/simsun.ttf", stopwords=stop_words)

    # generate word cloud.
    wc.generate(words)

    # store to file.
    wc.to_file('../cn_alice.png')

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# preprocessing_file('../cn.txt')
img_generate('../cn.txt', preprocessing_file)
