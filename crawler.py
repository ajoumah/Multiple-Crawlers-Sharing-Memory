"""Multiple Crawlers using Shared Memory.ipynb


# Topic-Spesific Crawler


---

By: Ahmad El Jouma - Yahya Tawil

Jan 2021

# 1- Installing Depenencies
"""

!pip install watchdog
!pip install -q tensorflow-text
!pip install -q tf-models-official
!apt-get install python3-dev default-libmysqlclient-dev
!pip install PyMySQL
!pip install Pattern

"""# 2- Download A Dataset to train our Classifier
[Two news article datasets, originating from BBC News](http://mlg.ucd.ie/datasets/bbc.html)
"""

!gsutil cp gs://dataset-uploader/bbc/bbc-text.csv .

"""# 3-Importing Packages"""

# Commented out IPython magic to ensure Python compatibility.
import csv
import ssl
import os
import itertools
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
# %matplotlib inline
import pandas as pd
from datetime import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
layers = keras.layers
models = keras.models

"""# 4- Crawler Settings"""

### Script Settings ###
conf = dict(
    base_url = 'https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html',
    number_of_threads = 4,
)
types_priority = {'business':3 ,'entertainment':2 ,'politics':5 ,'sport':4 ,'tech':1} # 1 is the highest ... 5 is the lowest
directory = r'Path/Crawler/Spider1/'
###

#Delete the Crawler Directory to start from a clean dirctory
if os.path.exists(directory) is True:
  print("Delete Directory")
  shutil.rmtree(directory)
if os.path.exists(directory) is False:
  print("Create Directory")
  os.mkdir(directory)

nltk.download('stopwords')  # download from NLTK a list of stopwords in English used later in tokenization
stop_words = stopwords.words('english')
#print(stop_words)

"""## 5-Training classifier network

### 5-1-Perpare the dataset to be used in training classifier network
"""

# Classifier was adapted from https://www.kaggle.com/yufengdev/bbc-text-categorization

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

data = pd.read_csv("bbc-text.csv")
#data.head()

train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))
train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)
#print(test_text)
max_words = 1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)

# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train tokens shape:', x_train.shape)
print('x_test tokens shape:', x_test.shape)
print('y_train one-hot representation shape:', y_train.shape)
print('y_test one-hot representation shape:', y_test.shape)

"""### 5-2- Build the classifier model, train and evaluation"""

# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 32
epochs = 2
drop_ratio = 0.5

# Build the model using Keras
model = models.Sequential() #Sequential: groups a linear stack of layers into
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(drop_ratio)) # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
model.add(layers.Dense(num_classes)) # A Dense layer feeds all outputs from the previous layer to all its neurons
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### 5-3- Testing the trained model (optional)"""

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()

#text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_text.iloc[i][:50], "...")
    print('Actual label:' + test_cat.iloc[i])
    print("Predicted label: " + predicted_label + "\n")

#samples = ["cristiano ronaldo was mad because of the last game he could not score"]
#tokenize.fit_on_texts(samples)
#sample_test = tokenize.texts_to_matrix(samples)
#prediction = model.predict(np.array(sample_test))
#print(text_labels)
#print(prediction)
#predicted_label = text_labels[np.argmax(prediction)]
#predicted_label

"""# 6- FolderWatcher / Watchdog"""

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
            preprocessing_classify()
        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print( "Received modified event - %s." % event.src_path)
            preprocessing_classify()

def print_something(path):
    print(path+"I am body of dog dog")

"""# 7- Priority Queue"""

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

links_to_crawl=MyPriorityQueue()

"""# 8- Downloader"""

regular_express = re.compile(r"https?://(\.)?")
class Crawler(threading.Thread):
    def __init__(self,base_url,have_visited, error_links,url_lock):

        threading.Thread.__init__(self)
        print(f"Web Crawler worker {threading.current_thread()} has Started")
        self.base_url = base_url
        self.have_visited = have_visited
        self.error_links = error_links
        self.url_lock = url_lock

    def run(self):
        my_ssl = ssl.create_default_context()
        my_ssl.check_hostname = False
        my_ssl.verify_mode = ssl.CERT_NONE

        while True:
            now = datetime.now()
            #timestamp = datetime.timestamp(now)
            # In this part of the code we create a global lock on our queue of
            # links so that no two threads can access the queue at same time
            self.url_lock.acquire(timeout=10)
            print(f"Queue Size: {links_to_crawl.qsize()}")
            link = links_to_crawl.get()


            # if the link is None the queue is exhausted or the threads are yet
            # process the links.

            if link is None:
                break

            # if The link is already visited we break the execution unless it passed 24h since last visit.

            find=[item for item in self.have_visited if link in item]
            if len(find)>0:
              find=tuple(find)
              if (now - find[0][1]).total_seconds() < 86400: # 86400 is 24H in secs, if the last visit time is less than 24h, then don't visit
                print(f"The link {link} is already visited")
                break
              else:
                  have_visited.remove(find[0]) #remove the record, to add new one later

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

                #if type(soup.find_all('a')) == 'NoneType':
                  #print("No links found in this page!")
                  #continue
                #else:
                  #for a_tag in soup.find_all('a'):
                      ##if (a_tag.get("href") not in self.have_visited) :
                      #if (a_tag.get("href").find('.html#') == -1): # do not crawl the intra document links
                          #self.links_to_crawl.put(a_tag.get("href"),1)
                      ##else:
                          ##print("Intra document link found!")

                      ##else:
                          ##print(f"The link {a_tag.get('href')} is already visited or is not part \
                          ##of the website")

                #print(f"Adding {link} to the crawled list at {timestamp}")
                self.have_visited.append((link,now))

                #print(have_visited)


            except URLError as e:
                #print(f"URL {link} threw this error {e.reason} while trying to parse")
                self.error_links.append(link)
            finally:
                links_to_crawl.task_done()

            self.url_lock.release()

"""# 9- Prepreocessing & Classification"""

preprocess_lock = 0
text_labels = encoder.classes_

def tag_visible(element): # to exclude the following tags content
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    if re.match(r"[\n]+",str(element)): return False
    return True

def preprocessing_classify():
  global preprocess_lock
  global text_labels
  #print("->Preprocessing()")
  if preprocess_lock == 0:
    preprocess_lock = 1
    for filename in os.listdir(directory):  #iterate with the files in the directory
            if filename.endswith(".html") : # select only the html files
                now = datetime.now()
                start_preprocessing_timestamp = datetime.timestamp(now)
                print(os.path.join(directory, filename))
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

                if len(post_text)>5:
                  class_input_text = [post_text]
                  tokenize.fit_on_texts(class_input_text)
                  temp = tokenize.texts_to_matrix(class_input_text)
                  prediction = model.predict(np.array(temp))
                  time.sleep(0.2)
                  print(text_labels) #['business' 'entertainment' 'politics' 'sport' 'tech']
                  print(prediction)
                  predicted_label = text_labels[np.argmax(prediction)]

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

                #build corpus and word counts from the uncleaned terms
                temp_mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
                temp_word_counts = [[(mydict[id], count) for id, count in line] for line in temp_mycorpus]

                mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in clean_lem_tokenized_list]
                word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]

                now = datetime.now()
                end_preprocessing_timestamp = datetime.timestamp(now)

                #if len(clean_lem_tokenized_list[-1]) is not 0: #generate csv file only if there are terms found
                if len(post_text)>5: #generate csv file only if there text in the webpage
                  with open(directory+'/'+filename+'.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['File Name = ',filename]) #write in csv page url
                    writer.writerow(['Raw = ',texts]) # found text
                    writer.writerow(tokenized_list[0]) # text after tokenization
                    writer.writerow(clean_tokenized_list) # text after removing the stop words
                    writer.writerow(clean_lem_tokenized_list[-1]) # text after cleaning and lemmatization
                    writer.writerow(['Exc Time = ',end_preprocessing_timestamp-start_preprocessing_timestamp]) # time needed to preporcess
                    writer.writerow(['Class = ',predicted_label])
                    writer.writerow(prediction)

                    if type(soup.find_all('a')) == 'NoneType':
                      print("No links found in this page!")
                    else:
                      for a_tag in soup.find_all('a'):
                        try:
                          if (a_tag.get("href").find('.html#') == -1): # do not crawl the intra document links
                           links_to_crawl.put(a_tag.get("href"),types_priority[predicted_label])
                        except:
                          print("No links found in this page!")

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
                os.rename(os.path.join(directory, filename),os.path.join(directory, filename)+"_r")
                continue
            else:
                continue
    preprocess_lock = 0

"""# Main"""

print("The Crawler has the following configuraions:")
print(conf)

base_url = conf["base_url"]
number_of_threads = conf["number_of_threads"]

url_lock = threading.Lock()
links_to_crawl.put(base_url,1)

have_visited = []
crawler_threads = []
error_links = []

before_starting = datetime.now()
for i in range(int(number_of_threads)):
    crawler = Crawler(base_url = base_url,
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

end_starting = datetime.now()
duration=(end_starting - before_starting).total_seconds()
minutes=duration/60
total=len(have_visited)/minutes
print(f"Total Number of pages visited are {len(have_visited)}")
print(f"Total Number of Errornous links: {len(error_links)}")
print(f"Total Time  in seconds is: {duration}")
print(f" Number of Pages in minutes: {total}")