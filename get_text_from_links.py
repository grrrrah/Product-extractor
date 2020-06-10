import urllib
from urllib.request import urlopen
from urllib.error import URLError
from bs4 import BeautifulSoup
import pandas as pd
from http.client import InvalidURL, RemoteDisconnected
import time


def return_df(source):
    df = pd.read_csv(source,sep=',',header=0,low_memory=False)
    df = df[['urls']]
    df = df.where(pd.notnull(df), 'None')
    return df


df_links = return_df(source='furniture stores pages.csv')
df_links = df_links[498:]
df_links = df_links.values.tolist()
df_links_ = [item for sublist in df_links for item in sublist]


def get_text(url):

    if 'http' in url:
        try:
            html = urllib.request.urlopen(url, timeout=20).read()
            soup = BeautifulSoup(html, features="html.parser")

            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            with open("{}.txt".format(str(int(time.time()))), "w", encoding='utf-8') as f:
                f.write(text)

            return text

        except (InvalidURL, URLError):
            pass


def get_all_texts(url_list):
    texts = [get_text(u) for u in url_list]
    return texts


txts = get_all_texts(url_list=df_links_)