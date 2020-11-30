import requests
from bs4 import BeautifulSoup
import time


def get_page(url, page_num):
    page = ''
    while page == '':
        try:
            page = requests.get(url)
            return page
        except:
            print('Page Num', str(page_num))
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue

BASE_URL = 'http://abcnotation.com'

for page_num in range(0, 850, 10):
    # Renaissance polyphony from Serpent Publications
    SEARCH_URL = 'http://abcnotation.com/searchTunes?q=site:serpent.serpentpublications.org&f=c&o=a&s={}'.format(page_num)


    # page = requests.get(SEARCH_URL)
    page = get_page(SEARCH_URL, page_num)
    soup = BeautifulSoup(page.content, 'html.parser')

    with open('Music_RNN/abc_classical2.txt', 'a+', 1) as abc_file:
        page = requests.get(SEARCH_URL, page_num)
        soup = BeautifulSoup(page.content, 'html.parser')
        # go to tune page with abc file text
        for hyperlinks in soup.findAll('a', attrs={'class':'label label-success'}):
            print(hyperlinks['href'])
            # page = requests.get(BASE_URL + hyperlinks['href'])
            page = get_page(BASE_URL + hyperlinks['href'], page_num)
            soup = BeautifulSoup(page.content, 'html.parser')
            for textarea in soup.findAll('textarea'):
                contents = BeautifulSoup(textarea.contents[0], features="html.parser").renderContents()
                print('Contents Type:', type(contents))
                print('Contents:', contents)
                contents_string = contents.decode("utf-8") 
                abc_file.write(contents_string + '\n')
