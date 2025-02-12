import requests
from bs4 import BeautifulSoup
import jieba
from wordcloud import WordCloud
from matplotlib import pyplot as plt

#網頁內文擷取
url = 'https://www.mropengate.com/2015/01/operating-system-ch3-processes.html'
response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, 'html.parser')

list_text = soup.find_all('li')
text_content = [li.get_text().strip() for li in list_text]

string_text = ''.join(text_content)
print(string_text)

#讀取停用字
with open('data/stops.txt', 'r', encoding = 'utf8') as f:
    stops = f.read().split('\n')
stops.append('\n')

#斷詞分析（檢查停用詞）
from collections import Counter
terms = []

for t in jieba.cut(string_text, cut_all = False):
    if t not in stops and t != ' ':
        terms.append(t)
        
sorted(Counter(terms).items(), key = lambda x:x[1], reverse = True)

#產生文字雲

import numpy as np 
from PIL import Image

cloud_mask = np.array(Image.open('data/cloud_mask.png'))
wc = WordCloud(background_color = 'white', max_words = 2000, mask = cloud_mask, font_path = 'data/simsun.ttf')
wc.generate_from_frequencies(Counter(terms))

wc.to_file(r'cloud.png')

plt.axis('off')
plt.imshow(wc, interpolation = 'bilinear')
