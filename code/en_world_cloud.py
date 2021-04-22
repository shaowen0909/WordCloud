from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Read the whole text.
with open('../en.txt', 'r') as file:
    text = file.read()

# read the mask image
alice_mask = np.array(Image.open('../alice_mask.png'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
               stopwords=stopwords, contour_width=3, contour_color='steelblue',
               scale=2)

# generate word cloud
wc.generate(text)

# store to file
wc.to_file('../en_alice.png')

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.show()
