"""Threading with Queue and Preprocessing.ipynb

"""

!pip install watchdog

!apt-get install python3-dev default-libmysqlclient-dev

!pip install PyMySQL

pip install Pattern


# Authors: Ahmed El Jouma 
# Dec 2020

import csv
import ssl
import os
import shutil
from urllib.request import Request, urlopen, URLError, urljoin
from urllib.parse import urlparse
import time
import threading
import queue
from bs4 import BeautifulSoup
from bs4.element import Comment
from pathlib import Path
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from queue import PriorityQueue
import gensim
from gensim import corpora
from pprint import pprint
from gensim.utils import simple_preprocess
from gensim.utils import simple_preprocess, lemmatize
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from nltk.corpus import stopwords
from multiprocessing import cpu_count
from smart_open import smart_open
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


### Script Settings ###
conf = dict(
    base_url = 'https://docs.python.org/3/contents.html',
    number_of_threads = 2,
)
directory = r'Path/Crawler/Spider1/'
###

if os.path.exists(directory) is True:
  print("Delete Directory")
  shutil.rmtree(directory)
if os.path.exists(directory) is False:
  print("Create Directory")
  os.mkdir(directory)

nltk.download('stopwords')  # download from NLTK a list of stopwords in English
stop_words = stopwords.words('english')
#print(stop_words)


def tag_visible(element): # to exclude the following tags content
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    if re.match(r"[\n]+",str(element)): return False
    return True

def preprocessing():
  #print("->Preprocessing()")
  for filename in os.listdir(directory):  #iterate with the files in the directory
          if filename.endswith(".html") : # select only the html files
              #print(os.path.join(directory, filename))
              soup = BeautifulSoup(open(os.path.join(directory, filename)), 'html.parser')
              texts = soup.findAll(text=True) # select all text in the html file
              #print(texts)
              visible_texts = filter(tag_visible, texts)
              text = u",".join(t.strip() for t in visible_texts)
              text = text.lstrip().rstrip()
              text = text.split(',')
              post_text = ''
              for sen in text:
                if sen:
                  if len(sen.split(' '))>5: # Accept only texts with more than 20 words
                    sen = sen.rstrip().lstrip()
                    post_text += sen+' '
              #print(post_text)

              tokenized_list = [simple_preprocess(doc) for doc in [post_text]]
              clean_tokenized_list = [] #tokenized
              clean_lem_tokenized_list = [] #tokenized & lemmatization
              doc_out = []
              for token in tokenized_list[0]:
                  if token not in stop_words:  # remove stopwords
                          clean_tokenized_list.append(token)
                          lemmatized_word = lemmatize(str(token), allowed_tags=re.compile('(VV|NN|JJ|RB)'))  # lemmatize
                          if lemmatized_word:
                              doc_out = doc_out + [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
                          else:
                              continue
              clean_lem_tokenized_list.append(doc_out)

              #print (tokenized_list)
              #print (clean_tokenized_list)
              #print (clean_lem_tokenized_list[-1])

              # Create dictionary
              dictionary = corpora.Dictionary(clean_lem_tokenized_list)
              mydict = corpora.Dictionary()

              if len(clean_lem_tokenized_list[-1]) is not 0: #generate csv file only if there are terms found
                with open(directory+'/'+filename+'.csv', 'w', newline='') as file:
                  writer = csv.writer(file)
                  writer.writerow(tokenized_list[0])
                  writer.writerow(clean_tokenized_list)
                  writer.writerow(clean_lem_tokenized_list[-1])

              #build corpus and word counts from the uncleaned terms
              temp_mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
              temp_word_counts = [[(mydict[id], count) for id, count in line] for line in temp_mycorpus]

              mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in clean_lem_tokenized_list]
              word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]

              #print(dictionary.token2id)
              #pprint(mycorpus)
              #pprint(word_counts)

              #df = pd.DataFrame(word_counts[0], columns=['term', 'frequency'])
              #ax = df.plot(kind='bar', x='term',title=filename,figsize=(20,5))

              #df2 = pd.DataFrame(temp_word_counts[0], columns=['term', 'frequency'])
              #ax2 = df2.plot(kind='bar', x='term',title="Not cleaned -"+filename,figsize=(30,5))

              #fig = ax.get_figure()
              #fig.savefig(filename+'.svg')
              #fig2 = ax2.get_figure()
              #fig2.savefig("Not cleaned -"+filename+'.svg')

              # Download dataset
              #dataset = api.load("text8")
              #data = [d for d in dataset]
              # Split the data into 2 parts. Part 2 will be used later to update the model
              #data_part1 = data[:1000]
              #data_part2 = data[1000:]
              # Train Word2Vec model. Defaults result vector size = 100
              #model = Word2Vec(data_part1, min_count = 0, workers=cpu_count())
              #model.most_similar('topic')
              os.rename(os.path.join(directory, filename),os.path.join(directory, filename)+"_r")
              continue
          else:
              continue





###########################################################################################
class Watcher:

    def __init__(self,name,Directory):
        self.observer = Observer()
        self.thread_name=name
        self.DIRECTORY_TO_WATCH=Directory
    def run(self):
        event_handler = Handler()

        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH,recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
                #print('thread name is %s'  %(self.thread_name))

        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)
            preprocessing()
        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print( "Received modified event - %s." % event.src_path)
            preprocessing()

def print_something(path):
    print(path+"I am body of dog dog")


######################################################################################
class MyPriorityQueue(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self.counter = 0

    def put(self, item, priority):
        PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1

    def get(self, *args, **kwargs):
        _, _, item = PriorityQueue.get(self, *args, **kwargs)
        return item
######################################################################################


regular_express = re.compile(r"https?://(\.)?")
class Crawler(threading.Thread):
    def __init__(self,base_url, links_to_crawl,have_visited, error_links,url_lock):

        threading.Thread.__init__(self)
        print(f"Web Crawler worker {threading.current_thread()} has Started")
        self.base_url = base_url
        self.links_to_crawl = links_to_crawl
        self.have_visited = have_visited
        self.error_links = error_links
        self.url_lock = url_lock

    def run(self):
        my_ssl = ssl.create_default_context()
        my_ssl.check_hostname = False
        my_ssl.verify_mode = ssl.CERT_NONE

        while True:
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            # In this part of the code we create a global lock on our queue of
            # links so that no two threads can access the queue at same time
            self.url_lock.acquire(timeout=5)
            print(f"Queue Size: {self.links_to_crawl.qsize()}")
            link = self.links_to_crawl.get()
            self.url_lock.release()

            # if the link is None the queue is exhausted or the threads are yet
            # process the links.

            if link is None:
                break

            # if The link is already visited we break the execution unless it passed 24h since last visit.
            links=[]
            visits_time=[]
            for lnk,vis_time in self.have_visited:
              links.append(lnk)
              visits_time.append(vis_time)

            if link in links:
              if (have_visited[self.have_visited.index(link)][1])+ 86400 > timestamp:# 86400 is 24H in secs, if the last visit time is less than 24h, then don't visit
                print(f"The link {link} is already visited")
                break
              else:
                del have_visited[self.have_visited.index(link)] #remove the record, to add new one later

            try:

                link = urljoin(self.base_url,link)
                req = Request(link, headers= {'User-Agent': 'Mozilla/5.0'})
                response = urlopen(req, context=my_ssl)

                print(f"The URL {response.geturl()} crawled with \
                      status {response.getcode()}")
                addres=regular_express.sub('', response.geturl()).strip().strip('/')
                addres=addres.replace('/','.')

                soup = BeautifulSoup(response.read(),"html.parser")
                filename=addres+'.html'
                with open(directory+filename,'w') as f:
                    text=soup.prettify()
                    f.write(text)

                for a_tag in soup.find_all('a'):
                    #if (a_tag.get("href") not in self.have_visited) :
                    if (a_tag.get("href").find('.html#') == -1): # do not crawl the intra document links
                        self.links_to_crawl.put(a_tag.get("href"),1)
                    #else:
                        #print("Intra document link found!")

                    #else:
                        #print(f"The link {a_tag.get('href')} is already visited or is not part \
                        #of the website")

                print(f"Adding {link} to the crawled list at {timestamp}")
                self.have_visited.append((link,timestamp))

                print(have_visited)


            except URLError as e:
                print(f"URL {link} threw this error {e.reason} while trying to parse")
                self.error_links.append(link)
            finally:
                self.links_to_crawl.task_done()

print("The Crawler is started")
#base_url = input("Please Enter Website to Crawl > ")
#number_of_threads = input("Please Enter number of Threads > ")
#Settings file  (root URL and threading number)

print(conf)
base_url = conf["base_url"]
number_of_threads = conf["number_of_threads"]

#links_to_crawl = queue.Queue()
links_to_crawl=MyPriorityQueue()
url_lock = threading.Lock()
links_to_crawl.put(base_url,1)
                #Task(text)
have_visited = [] #TODO: add time of visiting, 24H
crawler_threads = []
error_links = []
#base_url, links_to_crawl,have_visited, error_links,url_lock
for i in range(int(number_of_threads)):
    crawler = Crawler(base_url = base_url,
                      links_to_crawl= links_to_crawl,
                      have_visited= have_visited,
                      error_links= error_links,
                      url_lock=url_lock)

    crawler.start()
    crawler_threads.append(crawler)

w = Watcher("my thread",directory)
    #w.run()
thread1 = threading.Thread(target = w.run())

thread1.start()
crawler_threads.append(thread1)
for crawler in crawler_threads:
    crawler.join()



print(f"Total Number of pages visited are {len(have_visited)}")
print(f"Total Number of Errornous links: {len(error_links)}")