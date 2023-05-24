#from dataloader import dataset2dataloader
from torch.optim import Adam
import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np
import os
import pandas as pd
import spacy
from torch.nn import init
from torchtext.legacy import data

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags, word_vectors=None, device="cpu"):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=word_vectors).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True).to(device)
        self.hidden2tag = nn.Linear(hidden_size*2, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True).to(device)

    def get_emissions(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embed(x)
        h0, c0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device), torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        lstm_out, (_, _) = self.lstm(embedded, (h0, c0))
        emissions = self.hidden2tag(lstm_out)
        return emissions
    
    def forward(self, x, y, mask):
        emissions = self.get_emissions(x)
        loss = -self.crf(emissions=emissions, tags=y, mask=mask)
        return loss

    def predict(self, x, mask=None):
        emissions = self.get_emissions(x)
        preds = self.crf.decode(emissions, mask)
        return preds
    
def prepare_data(dataset_path, debug=False):
    train_file_path = os.path.join(dataset_path, "train.txt")
    dev_file_path = os.path.join(dataset_path, "dev.txt")
    test_file_path = os.path.join(dataset_path, "test.txt")

    def process_file(file_path, target_file_path):
        sents, tags = [], []
        with open(file_path, "r") as f:
            lines = f.readlines()
            sent, tag = [], []
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    sents.append(" ".join(sent))
                    tags.append(" ".join(tag))
                    sent, tag = [], []
                else:
                    splited = line.split(" ")
                    sent.append(splited[0])
                    tag.append(splited[-1])
            if len(sent) != 0:
                sents.append(" ".join(sent))
                tags.append(" ".join(tag))
        df = pd.DataFrame()
        df["sent"] = sents if not debug else sents[:100]
        df["tag"] = tags if not debug else tags[:100]
        df.to_csv(target_file_path, index=False)

    train_csv = os.path.join(dataset_path, "train.csv") if not debug else os.path.join(dataset_path, "train_small.csv")
    dev_csv = os.path.join(dataset_path, "dev.csv") if not debug else os.path.join(dataset_path, "train_dev.csv")
    test_csv = os.path.join(dataset_path, "test.csv") if not debug else os.path.join(dataset_path, "train_test.csv")

    if not os.path.exists(test_csv):
        process_file(train_file_path, train_csv)
        process_file(dev_file_path, dev_csv)
        process_file(test_file_path, test_csv)

    return train_csv, dev_csv, test_csv

def dataset2dataloader(dataset_path="/data/wyf/InformationRetrievalProject/data/", batch_size=3, debug=False):
    train_csv, dev_csv, test_csv = prepare_data(dataset_path, debug=debug)

    def tokenizer(text):
        return text.split(" ")

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=False)
    TAG = data.Field(sequential=True, tokenize=tokenizer, lower=False)
    train, val, test = data.TabularDataset.splits(
        path='', train=train_csv, validation=dev_csv, test=test_csv, format='csv', skip_header=True,
        fields=[('sent', TEXT), ('tag', TAG)])

    TEXT.build_vocab(train, vectors='glove.6B.50d')  # , max_size=30000)
    TAG.build_vocab(val)
    # 下面不注释acc0.89 反之0.95
    #TAG.build_vocab(test)
    
    TEXT.vocab.vectors.unk_init = init.xavier_uniform

    DEVICE = "cpu"
    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.sent), device=DEVICE)
    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.sent), device=DEVICE)
    #test_iter = data.BucketIterator(test, batch_size=batch_size, sort_key=lambda x: len(x.sent), device=DEVICE)
    test_iter = data.Iterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

    return train_iter, val_iter, test_iter, TEXT.vocab, TAG.vocab

train_iter, val_iter, test_iter, sent_vocab, tag_vocab = dataset2dataloader(batch_size=128)
word_vectors = sent_vocab.vectors

#device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = torch.device('cpu')

model = BiLSTM_CRF(vocab_size=len(sent_vocab.stoi), embedding_dim=50, hidden_size=128, num_tags=len(tag_vocab.stoi), word_vectors=word_vectors, device=device)

epoch = 10
learning_rate = 0.01
model_path = "model_BC.pkl"

optimizer = Adam(model.parameters(), lr=learning_rate)

if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    for ep in range(epoch):
        model.train()
        for i, batch in enumerate(train_iter):
            x, y = batch.sent.t(), batch.tag.t()
            mask = (x != sent_vocab.stoi["<pad>"])
            optimizer.zero_grad()
            loss = model(x, y, mask)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch:{ep}, iter:{i}, loss:{loss.item()}", end=" ")

        model.eval()
        train_accs = []
        preds, golds = [], []
        for i, batch in enumerate(train_iter):
            x, y = batch.sent.t(), batch.tag.t()
            mask = (x != sent_vocab.stoi["<pad>"])
            with torch.no_grad():
                preds = model.predict(x, mask)
            right, total = 0, 0
            for pred, gold in zip(preds, y):
                right += np.sum(np.array(pred) == gold[:len(pred)].numpy())
                total += len(pred)
            train_accs.append(right*1.0/total)
        train_acc = np.array(train_accs).mean()

        val_accs = []
        for i, batch in enumerate(val_iter):
            x, y = batch.sent.t(), batch.tag.t()
            mask = (x != sent_vocab.stoi["<pad>"])
            with torch.no_grad():
                preds = model.predict(x, mask)
            right, total = 0, 0
            for pred, gold in zip(preds, y):
                right += np.sum(np.array(pred) == gold[:len(pred)].numpy())
                total += len(pred)
            val_accs.append(right * 1.0 / total)
        val_acc = np.array(val_accs).mean()
        print("epoch %d train acc:%.2f, val acc:%.2f" % (ep, train_acc, val_acc))
torch.save(model, model_path)

model.eval()
test_accs = []
for i, batch in enumerate(test_iter):
    x, y = batch.sent.t(), batch.tag.t()
    mask = (x != sent_vocab.stoi["<pad>"])
    with torch.no_grad():
        preds = model.predict(x, mask)
    right, total = 0, 0
    for pred, gold in zip(preds, y):
        right += np.sum(np.array(pred) == gold[:len(pred)].numpy())
        total += len(pred)
    test_accs.append(right * 1.0 / total)
test_acc = np.array(test_accs).mean()
print("test acc:%.2f" % (test_acc))

#test_sents = ["My name is Yufei Wang , I am from Jinzhou , Hubei , China ."]
#test_sents = ["HUST is the abbreviation of Huazhong University of Science and Technology . It is located in Hongshan , Wuhan , China ."]
test_sents = ["Sufjan Stevens released his thirteenth album Carrie and Lowell on March 31 , 2015 in America and received a high score of 9 . 3 from Pitchfork ."]

for sent in test_sents:
    ids = [sent_vocab.stoi[word] for word in sent.split(" ")]
    input_tensor = torch.tensor([ids])
    mask = input_tensor != sent_vocab.stoi["<pad>"]
    with torch.no_grad():
        pred = model.predict(input_tensor, mask)
        
print(sent, "\n", [tag_vocab.itos[tag_id] for tag_id in pred[0]])