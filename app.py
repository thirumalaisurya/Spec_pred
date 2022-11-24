import re
import uuid
import nltk
import pickle
import json
import pickle
import requests
import subprocess
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from flask import Flask,url_for,request,jsonify
from nltk.tokenize import sent_tokenize, word_tokenize
#from preprocessing_pipeline import text_cleaner
#from config import icliniq_stopwords
from flashtext import KeywordProcessor
#import spam_identifier
#from predict_symptom import predict_disease
from nltk.stem import PorterStemmer,SnowballStemmer,WordNetLemmatizer
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, redirect, url_for,send_from_directory
#from werkzeug.utils import secure_filename
#from responsetime import MSKCC_Analyzer,MSKCC_Analyzer_v2
#from dataloader import summary_time_calculator
#from dataloader import column_checker
#from dataloader import dataloader_from_S3
#from datapipeline import my_dic_ad
from flashtext import KeywordProcessor
pipeline = load(open("E:/TEST/text_classification.joblib",'rb'))

#---------- Create Flask Instance-------------#
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def hello():
    return 'iCliniq Advertisement API and iCliniq smartAPI !!'


@app.route('/predict',methods=['GET','POST'])
def predict():
    main_data = request.get_json()
    query = main_data['query']
    #predict_disease(query)
    sample_x = [query]
    #sample_x = np.array(sample_x).reshape(1,len(sample_x))
    print('Predicted Speciality: ',pipeline.predict(sample_x)[0])
    json_obj = json.dumps(pipeline.predict(sample_x)[0], indent=4)
        #output = gbm.predict(sample_x[0])
        #predict_disease0 = mod.predict_proba([sample_x][0])

            #pred0 = icliniqML.predict_proba([data])
        #for label in pred:
       # predict_disease0 = np.array(predict_disease).reshape(1,len(predict_disease))
        #output = ""
        #print(output)
        #print(sample_x)
    return json_obj
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False, port = 4567)
