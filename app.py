from flask import Flask,render_template,request, url_for, flash, redirect
from flask_restful import Api, Resource,reqparse
from flask_cors import CORS
import os

import torch as T
import pandas as pd
import torch.nn as N
import torch.optim as O
import datetime
import pytz
import timeago

from config import *

wordseg_model = WordsegModel(dim_charvec=32, dim_trans=256, no_layers=4).to(device=device)
#ใส่ path model
wordseg_model.load_state_dict(T.load('word_segmodel_bigru_256_no4.pt',map_location=device))
wordseg_model.eval()

app = Flask(__name__)
api = Api(app)
CORS(app)
app.config['SECRET_KEY'] = 'db48d9283c4a3464559ac1efb6f5c111242a95d3c98455ec'

message = []
transection_checked = reqparse.RequestParser()
transection_checked.add_argument("txt",required=True, type=str, help="Required str txt")

message = [{'title': 'ตัดคำ', 'word': ['ลอง', 'ตัด', 'คำ', 'ได้', 'ดี', 'มาก'], 'time': datetime.datetime(2022, 4, 30, 18, 2, 29, 849566), 'len': 6, 'raw': 'ลองตัดคำได้ดีมาก'}]

@app.route('/',methods=('GET', 'POST'))
def hello():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        content = content.replace("\n","").replace("\r","")
        if not title:
            flash('Title is required!')
        elif not content:
            flash('Content is required!')
        else:
            words = tokenize(wordseg_model, content)
            words = list(map(lambda x:''.join(x),words))
            words = "|".join(words).replace(' ','').split('|')
            message.append({"title":title,"word":words,"time":datetime.datetime.now(),"len":len(words),"raw":content})
            print(message)
            return redirect(url_for('hello'))

    now = datetime.datetime.now()
    mg = message.copy()
    for i in range(len(message)):
        message[i]['timelong'] = timeago.format(message[i]['time'], now)

    return render_template('create.html',comments=mg[::-1])

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.route('/tokenize',methods=['POST'])
def word():
    args = transection_checked.parse_args()
    # print(args['msg'])
    words = tokenize(wordseg_model, args['txt'])
    words = list(map(lambda x:''.join(x),words))
    words = "|".join(words).replace(' ','').split('|')
    return {"words":words,"size":len(words)}

@app.route('/result/')
def result():
    now = datetime.datetime.now()
    mg = message.copy()
    for i in range(len(message)):
        message[i]['timelong'] = timeago.format(message[i]['time'], now)
    return render_template('result.html',comments=mg[::-1])



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=3000)
