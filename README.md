# 📚 Topic-Specific Web Crawler  
### A Study of Existing Architectures and Possible Improvements  
**Authors:** Ahmad El Jouma & Yahya Tawil  
**Date:** April 14, 2021  

---

## 🔍 Project Overview  
This project investigates web crawlers, focusing on **topic-specific (focused) web crawlers** designed to download and index relevant web pages only, optimizing crawling efficiency and bandwidth usage. It includes both a study of existing architectures and a design & implementation of a topic-specific web crawler leveraging machine learning for classification.

---

## 📝 Abstract  
Web crawlers are critical to search engines, systematically browsing and downloading web pages to build indexed databases. This study explores web crawler architectures and proposes improvements by implementing a topic-specific crawler. This crawler fetches and indexes web content relevant to particular topics using classification algorithms, aiming to optimize crawling efficiency and reduce unnecessary data downloads.

---

## 🚀 Features & Highlights  

- **Seed URL Initialization:** Starts crawling from prioritized seed URLs.  
- **Multithreaded Crawling:** Multiple spiders work concurrently to increase crawling speed.  
- **Priority-Based URL Queue:** High priority URLs are crawled first.  
- **Politeness:** Requests emulate browser headers to prevent server blocking.  
- **Folder Watchdog:** Monitors downloaded pages folder and triggers preprocessing.  
- **Visited Pages Management:** Avoid revisiting pages within a 24-hour threshold.  
- **Text Preprocessing:** Tokenization, stop word removal, lemmatization, and Bag of Words creation.  
- **Machine Learning Classification:** Uses a fully connected neural network for topic classification to decide whether to crawl linked URLs.

---

## 📖 Table of Contents  

- [Introduction](#-introduction)  
- [Crawler Types & Strategies](#-crawler-types--strategies)  
- [Crawler Design](#-crawler-design)  
- [Implementation Details](#-implementation-details)  
- [Machine Learning Classifier](#-machine-learning-classifier)  
- [Usage](#-usage)  
- [Contact](#-contact)  

---

## 📚 Introduction  

Data availability on the web is growing rapidly. Search engines rely on crawlers to navigate and index web pages efficiently. This project focuses on **topic-specific crawlers** that selectively fetch pages relevant to predefined subjects, saving bandwidth and computational resources.

---

## 🕸️ Crawler Types & Crawling Strategies  

- **Universal Crawlers:** Crawl all pages without specialization.  
- **Topical Crawlers:** Focus on specific domains or topics.  
- **Incremental Crawlers:** Crawl only updated or changed pages.  
- **Deep Web Crawlers:** Target pages hidden behind query forms.

### Crawling Strategies:  

- **Depth-First Search (DFS):** Vertically traverse links deep down before backtracking.  
- **Breadth-First Search (BFS):** Horizontally scan all sibling links before going deeper.  
- **Best-First Search:** Uses heuristics to prioritize URLs based on relevance.

---

## 🏗️ Crawler Design  

### System Architecture includes:  

- **Scheduler:** Manages download timings and URL queue.  
- **Downloader:** Fetches web pages, mimics browser requests to avoid blocks.  
- **Parser & Extractor:** Extracts text and URLs from downloaded pages.  
- **Preprocessing:** Cleans and tokenizes text, removes irrelevant tags and stopwords.  
- **Feature Extraction:** Encodes text into numeric format for ML input.  
- **Classifier:** Neural network predicts page relevance to topics, guiding crawler decisions.

---

## ⚙️ Implementation Details  

- **Downloader:** Uses HTTP requests with user-agent spoofing for polite crawling.  
- **Folder Watchdog:** Monitors downloaded folder and triggers preprocessing on new files.  
- **URL Queue Management:** Thread-safe locking ensures no duplicate crawling.  
- **Visited Pages:** URLs visited are timestamped and revisited after 24 hours for freshness.  
- **Multithreading:** Multiple spiders crawl URLs concurrently with queue locking and timeout mechanisms.

---

## 🧠 Machine Learning Classifier  

- **Architecture:** Fully connected neural network with input size based on dictionary size.  
- **Layers:**  
  - Input Layer  
  - Dense Layer (512 units, ReLU)  
  - Dropout (50%)  
  - Dense Layer (512 units)  
  - Output Layer (5 classes with Softmax)  
- **Purpose:** Classify web pages into topic categories to prioritize crawling relevant pages.

---

## 🚀 Usage  

1. **Initialize seed URLs** with priorities in the URL list.  
2. **Run the crawler**, which will start fetching pages using multiple threads.  
3. **Monitor the output folder** for downloaded pages and classification results.  
4. **Adjust seed URLs and classification thresholds** as needed for targeted crawling.

---

## 📬 Contact  

**Ahmed El Jouma**  
Hasan Kalyoncu University  
Email: ael.jouma@std.hku.edu.tr  

**Yahya Tawil**  
Hasan Kalyoncu University  
Email: yahya.tawil@std.hku.edu.tr  

---


