# Sentiment Analyzer Chrome Extension
> Course Project for "CSE 256 - Statisical NLP" at UCSD.

A sentiment analyzer chrome extension utilizing SWEM (simple word embedding models). When you select some text in a webpage perform an analysis in right-click menu, you will be able to see a score indicating your tone (positivity) in the extension tab.

## Requirements
- Pytorch
- Python 3.7
- spacy
- nltk
- torchtext

## Usage
#### 0. Train the model (if you need to)
```sh
cd model
python train.py
```
#### 1. Set up the flask RESTFUL server
```sh
cd model
set FLASK_APP=api.py
flask run
```
#### 2. Load the extension (demo folder) in developer mode in Chrome
#### 3. Use the extension