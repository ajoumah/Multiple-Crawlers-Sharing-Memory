"""Threading with Queue

"""

!pip install watchdog

!apt-get install python3-dev default-libmysqlclient-dev

!pip install PyMySQL

pip install Pattern

# Authors: Ahmed El Jouma 
# Dec 2020
# Refs: In Project Report

from urllib.request import Request, urlopen, URLError, urljoin
from urllib.parse import urlparse
import time
import threading
import queue
from bs4 import BeautifulSoup
from bs4.element import Comment
import ssl
import os
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
from nltk.corpus import stopwords
from multiprocessing import cpu_count
import gensim.downloader as api
from smart_open import smart_open
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
###################################
from datetime import date, time, datetime
from datetime import datetime, timedelta
NUMBER_OF_SECONDS = 86400 # seconds in 24 hours
###################################
directory = r'PathCrawler/Spider1'
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
  print("->Preprocessing()")
  for subdir, dirs, files in os.walk(directory):
    #print (dirs)
    for dir in dirs:
      complete_dir = directory+'/'+dir
      for filename in os.listdir(complete_dir):  #iterate with the files in the directory
          if filename.endswith(".html") : # select only the html files
              #print(os.path.join(complete_dir, filename))
              soup = BeautifulSoup(open(os.path.join(complete_dir, filename)), 'html.parser')
              texts = soup.findAll(text=True) # select all text in the html file
              visible_texts = filter(tag_visible, texts)
              text = u",".join(t.strip() for t in visible_texts)
              text = text.lstrip().rstrip()
              text = text.split(',')
              post_text = ''
              for sen in text:
                if sen:
                  if len(sen.split(' '))>20: # Accept only texts with more than 20 words
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
                          lemmatized_word = lemmatize(str(token), allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
                          if lemmatized_word:
                              doc_out = doc_out + [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
                          else:
                              continue
              clean_lem_tokenized_list.append(doc_out)

              print (tokenized_list)
              #print (clean_tokenized_list)
              #print (clean_lem_tokenized_list[-1])

              # Create dictionary
              dictionary = corpora.Dictionary(clean_lem_tokenized_list)
              mydict = corpora.Dictionary()

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
              os.rename(os.path.join(complete_dir, filename),os.path.join(complete_dir, filename)+"_r")
              continue
          else:
              continue





###########################################################################################
class Watcher:
    #DIRECTORY_TO_WATCH = "Pathwatcher"


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
firstFolder='PathCrawler/Spider1/'

secondFolder='PathCrawler/Spider2/'
regular_express = re.compile(r"https?://(\.)?")
iteration=0
class Crawler(threading.Thread):
    def __init__(self,base_url, links_to_crawl,have_visited, error_links,url_lock,itera):

        threading.Thread.__init__(self)
        print(f"Web Crawler worker {threading.current_thread()} has Started")
        self.base_url = base_url
        self.links_to_crawl = links_to_crawl
        self.have_visited = have_visited
        self.error_links = error_links
        self.url_lock = url_lock
        self.itera=itera
    def run(self):

        my_ssl = ssl.create_default_context()


        my_ssl.check_hostname = False


        my_ssl.verify_mode = ssl.CERT_NONE



        while True:

            # In this part of the code we create a global lock on our queue of
            # links so that no two threads can access the queue at same time
            self.url_lock.acquire()
            print(f"Queue Size: {self.links_to_crawl.qsize()}")
            link = self.links_to_crawl.get()
            self.url_lock.release()

            # if the link is None the queue is exhausted or the threads are yet
            # process the links.

            if link is None:
                break

            # if The link is already visited we break the execution.
            if link in self.have_visited:
                print(f"The link {link} is already visited")
                break

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
                with open(firstFolder+filename,'w') as f:
                    text=soup.prettify()
                    print("I am before cleaning")
                    #print(text)
                    f.write(text)

                for a_tag in soup.find_all('a'):

                    if (a_tag.get("href") not in self.have_visited)  : #and (self.itera <1000)
                        self.links_to_crawl.put(a_tag.get("href"),1)
                        self.itera=self.itera+1
                    else:
                        print(f"The link {a_tag.get('href')} is already visited or is not part \
                        of the website")

                print(f"Adding {link} to the crawled list")
                self.have_visited.add(link)

            except URLError as e:
                print(f"URL {link} threw this error {e.reason} while trying to parse")

                self.error_links.append(link)

            finally:
                self.links_to_crawl.task_done()

print("The Crawler is started")
base_url = input("Please Enter Website to Crawl > ")
number_of_threads = input("Please Enter number of Threads > ")

#links_to_crawl = queue.Queue()
links_to_crawl=MyPriorityQueue()
url_lock = threading.Lock()
links_to_crawl.put(base_url,1)
                #Task(text)
have_visited = set()
crawler_threads = []
error_links = []
#base_url, links_to_crawl,have_visited, error_links,url_lock
before_starting = datetime.now()
iteration=0
for i in range(int(number_of_threads)):
    crawler = Crawler(base_url = base_url,
                      links_to_crawl= links_to_crawl,
                      have_visited= have_visited,
                      error_links= error_links,
                      url_lock=url_lock,itera=iteration)

    crawler.start()
    crawler_threads.append(crawler)

w = Watcher("my thread","PathCrawler/Spider1/")
    #w.run()
thread1 = threading.Thread(target = w.run())

thread1.start()
crawler_threads.append(thread1)
#for crawler in crawler_threads:
#    crawler.join()
for crawler in crawler_threads:
    crawler.join(30)
for crawler in crawler_threads:
    crawler.join()

# Thread can still be alive at this point. Do another join without a timeout
# to verify thread shutdown.
#t.join()

end_starting = datetime.now()
duration=(end_starting - before_starting).total_seconds()
minutes=duration/60
total=len(have_visited)/minutes
print(f"Total Number of pages visited are {len(have_visited)}")
print(f"Total Number of Errornous links: {len(error_links)}")
print(f"Total Time  in seconds is: {duration}")
print(f" Number of Pages in minutes: {total}")

