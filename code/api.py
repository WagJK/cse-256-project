from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from .model import SWEM_avg, SWEM_max, SWEM_hier, RNN, CNN
import json
import os
import random
import time
import spacy
import argparse

app = Flask(__name__) #create the Flask app
api = Api(app)
parser_flask = reqparse.RequestParser()

# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--embeddingSize', type=int, default=300)
# parser.add_argument('--batchSize', type=int, default=256)
# parser.add_argument('--vocabSize', type=int, default=25000)
# parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--embedding', type=str, default="glove.840B.300d")
# parser.add_argument('--reg', type=float, default=0.0)
# parser.add_argument('--dropout', type=float, default=0.0)

# args = parser.parse_args()
# EMBEDDING_DIM = args.embeddingSize
# BATCH_SIZE = args.batchSize
# MAX_VOCAB_SIZE =args.vocabSize
# N_EPOCHS = args.epoch
# pretrainedEmb =args.embedding
# add_reg = args.reg
# add_dropout =args.dropout
# datafile = 'imdb' # imdb or sst

EMBEDDING_DIM = 300
BATCH_SIZE = 256
MAX_VOCAB_SIZE = 25000
N_EPOCHS = 10
pretrainedEmb = "glove.840B.300d"
add_reg = 0
add_dropout = 0
datafile = 'imdb' # imdb or sst


if os.path.exists('.data/'+datafile+'/valid.json'):
    TEXT = data.Field(include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)
    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path = '.data/' + datafile,
        train = 'train.json',
        validation = 'valid.json',
        test = 'test.json',
        format = 'json',
        fields = fields
    )
else:
    TEXT = data.Field(tokenize='spacy', include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)
    if datafile == 'imdb':
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    elif datafile == 'sst':
        train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral')
    train_examples = [vars(t) for t in train_data]
    valid_examples = [vars(t) for t in valid_data]
    test_examples = [vars(t) for t in test_data]
    with open('.data/'+datafile+'/train.json', 'w+') as f:
        for example in train_examples:
            json.dump(example, f)
            f.write('\n')
    with open('.data/'+datafile+'/valid.json', 'w+') as f:
        for example in valid_examples:
            json.dump(example, f)
            f.write('\n')
    with open('.data/'+datafile+'/test.json', 'w+') as f:
        for example in test_examples:
            json.dump(example, f)
            f.write('\n')


TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = pretrainedEmb,
                 unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(TEXT.vocab)
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = SWEM_avg(INPUT_DIM, EMBEDDING_DIM, PAD_IDX, add_dropout)
# model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, add_dropout, PAD_IDX)
# model = CNN(INPUT_DIM, EMBEDDING_DIM, 300, PAD_IDX, DROPOUT)

model1 = SWEM_avg(INPUT_DIM, EMBEDDING_DIM, PAD_IDX, add_dropout)
model2 = SWEM_hier(INPUT_DIM, EMBEDDING_DIM, PAD_IDX, add_dropout)
model1.load_state_dict(torch.load('avg_model_'+datafile+'.pt'))
model2.load_state_dict(torch.load('model_'+datafile+'.pt'))
nlp = spacy.load('en')

def predict_sentiment(sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    if length_tensor > 5:
        prediction = torch.sigmoid(model2(tensor, length_tensor))
    else:
        prediction = torch.sigmoid(model1(tensor, length_tensor))
    return prediction.item()

class Query(Resource):
    def post(self):
        parser_flask.add_argument('text', type=str)
        args = parser_flask.parse_args()
        print(args)
        return {
            'text': args['text'],
            'sentiment': predict_sentiment(args['text'])
        }
        

api.add_resource(Query, '/')
if __name__ == '__main__':
    app.run(debug=True)