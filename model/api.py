from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from .train import predict_sentiment
from .model import SWEM_avg, SWEM_max, SWEM_hier
import json
import os
import random
import time
import spacy

app = Flask(__name__) #create the Flask app
api = Api(app)
parser_flask = reqparse.RequestParser()

# load model
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

MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(
    train_data, 
    max_size = MAX_VOCAB_SIZE, 
    vectors = "glove.6B.300d", 
    unk_init = torch.Tensor.normal_
)
device = torch.device('cpu')

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = SWEM_avg(INPUT_DIM, EMBEDDING_DIM, PAD_IDX)
model.load_state_dict(torch.load('model_'+datafile+'.pt'))
nlp = spacy.load('en')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()


class Query(Resource):
    def post(self):
        parser_flask.add_argument('text', type=str)
        args = parser_flask.parse_args()
        print(args)
        return {
            'text': args['text'],
            'sentiment': predict_sentiment(model, args['text'])
        }

class Feedback(Resource):
    def post(self):
        parser_flask.add_argument('text', type=str)
        parser_flask.add_argument('sentiment', type=str)
        parser_flask.add_argument('feedback', type=str)
        args = parser_flask.parse_args()
        print(args)

        f = open('feedback.log', 'a')
        f.write(args['text'] + ' ' + args['sentiment'] + ' ' + args['feedback'])
        f.close()
        return "Feedback Received"
        

api.add_resource(Query, '/')
api.add_resource(Feedback, '/feedback')

if __name__ == '__main__':
    app.run(debug=True)