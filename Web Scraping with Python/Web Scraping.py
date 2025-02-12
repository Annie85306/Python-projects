import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://tw.buy.yahoo.com/category/4385983?sort=-sales'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
response = requests.get(url, headers = headers)

try:
    with open("lamp.html", "w", encoding = "utf-8") as f:
        f.write(response.text)
        print("讀取成功")
except:
    print("讀取失敗")

dict = {}
soup = BeautifulSoup(response.text, 'html.parser')
items = soup.find_all('div', class_ = 'sc-1drl28c-4 dkDIGK')
for i in items:
    name = i.find('span', class_ = "sc-dlyefy sc-gKcDdr sc-1drl28c-5 jHwfYO ikfoIQ jZWZIY")
    price = i.find('span', class_ = "sc-dlyefy sc-gKcDdr dfRcqf hFXgfs")
    if name and price:
        print(name.text.strip() + price.text.strip())
        name = name.text.strip()
        price = price.text.strip()
        dict[name] = price

df = pd.DataFrame(dict, index = [0])
df.to_excel("price_comparison.xlsx", index = False)
