import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from .model import SWEM_avg, SWEM_max, SWEM_hier
import json
import os
from torchtext import datasets
import random
import torch.optim as optim
import time
import spacy

SEED = 777
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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
TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.300d", 
                 unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    device = device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = SWEM_hier(INPUT_DIM, EMBEDDING_DIM, PAD_IDX)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_'+datafile+'.pt')
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('model_'+datafile+'.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


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