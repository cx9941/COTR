import requests
from bs4 import BeautifulSoup

# 发送请求获取网页内容
url = 'https://www.onetonline.org/find/all'  # 示例 URL
response = requests.get(url)
html_content = response.content

# 使用 BeautifulSoup 解析 HTML 内容
soup = BeautifulSoup(html_content, 'html.parser')

# 找到 class 为 'w-70 mw-10e sorter-text' 的所有 td 元素
td_elements = soup.find_all('td', class_='w-70 mw-10e sorter-text')

# 遍历所有找到的 td 元素，提取其中的链接
with open('link.txt', 'w') as w:
    for td in td_elements:
        a_tag = td.find('a', href=True)  # 查找包含 href 的 a 标签
        if a_tag:
            href = a_tag['href']  # 提取 href 链接
            print(a_tag.text, href)
            w.write(f"{a_tag.text}\t{href}\n")