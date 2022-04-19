from flask import Flask
from flask_restful import Api, Resource,reqparse
from flask_cors import CORS

import torch as T
import pandas as pd
# import pythainlp
import torch.nn as N
import torch.optim as O

device = T.device("cuda" if T.cuda.is_available() else "cpu")

idx2char = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xa0', '®', 'é', 'ü', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ฦ', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', '฿', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๎', '๏', '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙', '๚', '๛', '\u200e', '–', '—', '‘', '’', '…', '™']

char2idx={v:k for k,v in enumerate(idx2char)}



def str2idxseq(charseq):
    idxseq = []
    for char in charseq:
        char = char.lower()
        idxseq.append(char2idx.get(char,0))
        """if char in char2idx:
            idxseq.append(char2idx[char])
        else:
            idxseq.append(char2idx[None])"""
    return idxseq

def idxseq2str(idxseq):
    charseq = []
    for idx in idxseq:
        if idx < len(idx2char):
            charseq.append(idx2char[idx])
        else:
            charseq.append(' ')
    return charseq

def sent2data(sent):
    charidxs = []
    wordbrks = []
    for charseq in sent:
        idxs = str2idxseq(charseq)
        charidxs.extend(idxs)
        wordbrks.extend((len(idxs) - 1) * [False] + [True])
    return (T.tensor(charidxs, device=device), T.tensor(wordbrks, device=device))

def corpus2dataset(corpus):
    dataset = []
    for sent in corpus:
        charidxs, wordbrks = sent2data(sent)
        dataset.append((charidxs, wordbrks))
    return dataset

def corpus2dataset_dl(corpus):
    dataset = []
    data = []
    label = []
    for sent in corpus:
        charidxs, wordbrks = sent2data(sent)
        data.append(charidxs)
        label.append(wordbrks)
    return data , label

def wordbrks2brkvec(wordbrks):
    return wordbrks.bool().long()

class WordsegModel(N.Module):
    def __init__(self, dim_charvec, dim_trans, no_layers):
        super(WordsegModel, self).__init__()
        self._dim_charvec = dim_charvec
        self._dim_trans = dim_trans
        self._no_layers = no_layers
        
        self._charemb = N.Embedding(184, self._dim_charvec)
        
        self._rnn = N.GRU(
            self._dim_charvec, self._dim_trans, self._no_layers,
            batch_first=True, bidirectional=True ,dropout=0.2
        )

        self._tanh = N.Tanh()
        self._hidden = N.Linear(2 * self._dim_trans, 2)    # Predicting two classes: break / no break
        self._log_softmax = N.LogSoftmax(dim=1)
        
    def forward(self, charidxs):
        # try:
            charvecs = self._charemb(T.as_tensor(charidxs,device=device))
            # print('charvecs =\n{}'.format(charvecs))
            ctxvecs, lasthids = self._rnn(charvecs.unsqueeze(0))
            ctxvecs, lasthids = ctxvecs.squeeze(0), lasthids.squeeze(1)
            # print('ctxvecs =\n{}'.format(ctxvecs))
            statevecs = self._hidden(self._tanh(ctxvecs))
            # print('statevecs =\n{}'.format(statevecs))
            brkvecs = self._log_softmax(statevecs)
            # print('brkvecs =\n{}'.format(brkvecs))
            return brkvecs

wordseg_model = WordsegModel(dim_charvec=32, dim_trans=256, no_layers=4).to(device=device)
#ใส่ path model
wordseg_model.load_state_dict(T.load('word_segmodel_bigru_256.pt',map_location=device))
wordseg_model.eval()

def tokenize(wordseg_model, charseq):
    charidxs = str2idxseq(charseq)
    pred_brkvecs = wordseg_model(charidxs)
    # return pred_brkvecs
    pred_wordbrks = []
    for i in range(len(charidxs)):
        pred_wordbrk = (pred_brkvecs[i][0] < pred_brkvecs[i][1])
        # print(pred_wordbrk)
        pred_wordbrks.append(pred_wordbrk)
    
    sent = []
    word = []
    begpos = 0
    for i in range(len(pred_wordbrks)):
        if pred_wordbrks[i]:
            word.append(charseq[i])
            sent.append(word)
            word = []
            begpos = i
        else:
            word.append(charseq[i])
    if len(word) > 0: sent.append(word)
        
    return sent


app = Flask(__name__)
api = Api(app)
CORS(app)

transection_checked = reqparse.RequestParser()
transection_checked.add_argument("txt",required=True, type=str, help="Required str txt")

class wordseg(Resource):
    def get(self):
        #print("Call! wordseg")
        return 'word segment'

    def post(self):
        args = transection_checked.parse_args()
        print(args['txt'])
        words = tokenize(wordseg_model, args['txt'])
        words = list(map(lambda x:''.join(x),words))
        return words

api.add_resource(wordseg, "/")

if __name__ == '__main__':
    app.run(debug=True)
