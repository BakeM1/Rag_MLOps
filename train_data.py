import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import Normalizer
import yaml
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import os
import wget

#defining the url of the file
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv"

#filename = wget.download(url)

import subprocess
import sys



os.system('pip install boto3')

import boto3
import botocore

BUCKET_NAME = 'mlopsfilesdvc' # replace with your bucket name
KEY = 'haberman.csv' # replace with your object key

#s3 = boto3.resource('s3')

s3 = boto3.resource('s3',
         aws_access_key_id='AKIAWUFQFXJL6WAJHASG',
         aws_secret_access_key= 'Df/OBvp7FJeg6ZtM3X2keynE7w1412/VNEBuDehI')

filename=s3.Bucket(BUCKET_NAME).download_file(KEY, 'haberman.csv')
print(filename)

import pandas as pd
import numpy as np
from sklearn import preprocessing


filename = 'haberman.csv'
columns = ['age', 'year', 'node', 'class']
df = pd.read_csv(filename, header=None, names=columns)
labelEncoder = preprocessing.LabelEncoder()
#Encode the class column into 0 and 1
df['class'] = labelEncoder.fit_transform(df['class'])
df.to_csv("haberman_processed.csv")

filename2 = 'haberman_processed.csv'
df = pd.read_csv(filename2, index_col=0)

y = df.pop('class').to_numpy()

X = df.to_numpy()

X = Normalizer().fit_transform(X)

#Using a Logistic Regression in production
#clf = LogisticRegression(solver=yaml.safe_load(open('params.yaml'))['solver'])
#clf = MultinomialNB()
clf = LinearDiscriminantAnalysis()
y_pred = cross_val_predict(clf, X, y, cv = yaml.safe_load(open('params.yaml'))['cv'])

# Metrics for comparing performance bw models
acc = np.mean(y_pred==y)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

# Dumping the model into a serialized file
model_name = "final_model.sav"
pickle.dump(clf, open(model_name, 'wb'))
