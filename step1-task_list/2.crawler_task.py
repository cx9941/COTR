import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv('link.txt', sep='\t', header=None)
df.columns = ['occupation', 'href']

def extract_task(url):
    try:
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        td_elements = soup.find_all('div', id='Tasks')
        task_list = [i.text for i in td_elements[0].find_all('div', class_='order-2 flex-grow-1')]
        return task_list
    except Exception as e:
        print(url, e)
        return []

df['task'] = df['href'].progress_apply(extract_task)
df = df.explode('task')
df.to_csv('task_art.csv', index=None, sep='\t')