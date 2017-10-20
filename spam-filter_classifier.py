#Dataset from "https://www.cs.cmu.edu/~./enron/"


# Load required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import pickle
import seaborn as sns
import sys
sys.setrecursionlimit(1500)
%matplotlib inline

#Preparing Enron Data
#We will extract and load the Enron spam data in Pandas Dataframe.
#Enron data combined with Spam assasin dataset has been obtained from : https://www.cs.bgu.ac.il/~elhadad/nlp16/spam_classifier.html and I also #used their code to process the data into Pandas dataframe
#We will extract and load the Enron spam data in Pandas Dataframe.

def progress(i, end_val, bar_length=50):
    '''
    Print a progress bar of the form: Percent: [#####      ]
    i is the current progress value expected in a range [0..end_val]
    bar_length is the width of the progress bar on the screen.
    '''
    percent = float(i) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

NEWLINE = '\n'
HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('../data/enron//spam',        SPAM),
    ('../data/enron//easy_ham',    HAM),
    ('../data/enron//hard_ham',    HAM),
    ('../data/enron//beck-s',      HAM),
    ('../data/enron//farmer-d',    HAM),
    ('../data/enron//kaminski-v',  HAM),
    ('../data/enron//kitchen-l',   HAM),
    ('../data/enron//lokay-m',     HAM),
    ('../data/enron//williams-w3', HAM),
    ('../data/enron//BG',          SPAM),
    ('../data/enron//GP',          SPAM),
    ('../data/enron//SH',          SPAM)
]

SKIP_FILES = {'cmds'}
NEWLINE="\n"

def read_files(path):
    '''
    Generator of pairs (filename, filecontent)
    for all files below path whose name is not in SKIP_FILES.
    The content of the file is of the form:
        header....
        <emptyline>
        body...
    This skips the headers and returns body only.
    '''
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(l, path, classification):
    rows = []
    index = []
    for i, (file_name, text) in enumerate(read_files(path)):
        if ((i+l) % 100 == 0):
            progress(i+l, 58910, 50)
        rows.append({'text': text, 'label': classification,'file':file_name})
        index.append(file_name)
   
    data_frame = pd.DataFrame(rows, index=index)
    return data_frame, len(rows)

def load_data():
    data = pd.DataFrame({'text': [], 'label': [],'file':[]})
    l = 0
    for path, classification in SOURCES:
        data_frame, nrows = build_data_frame(l, path, classification)
        data = data.append(data_frame)
        l += nrows
    data = data.reindex(np.random.permutation(data.index))
    return data
# We will load the Email spam dataset into Panadas dataframe here . 
data=load_data()
new_index=[x for x in range(len(data))]
data.index=new_index

def token_count(row):
    'returns token count'
    text=row['tokenized_text']
    length=len(text.split())
    return length
#We will add two more columns to our dataframe for tokenized text and token count.
def tokenize(row):
    "tokenize the text using default space tokenizer"
    text=row['text']
    lines=(line for line in text.split(NEWLINE) )
    tokenized=""
    for sentence in lines:
        tokenized+= " ".join(tok for tok in sentence.split())
    return tokenized

data['tokenized_text']=data.apply(tokenize, axis=1)
data['token_count']=data.apply(token_count, axis=1)
data['lang']='en'
#Prepare training and test data
# We randomize the rows to subset the dataframe
df.reset_index(inplace=True)
df=df.reindex(np.random.permutation(df.index))
len_unseen=10000
df_unseen_test= df.iloc[:len_unseen]
df_model = df.iloc[len_unseen:]

print('total emails for unseen test data : ', len(df_unseen_test))
print('\t total spam emails for enron  : ', len(df_unseen_test[(df_unseen_test['lang']=='en') & (df_unseen_test['label']=='spam')]))
print('\t total normal emails for enron  : ', len(df_unseen_test[(df_unseen_test['lang']=='en') & (df_unseen_test['label']=='ham')]))
print()

print('total emails for model training/validation : ', len(df_model))
print('\t total spam emails for enron  : ', len(df_model[(df_model['lang']=='en') & (df_model['label']=='spam')]))
print('\t total normal emails for enron  : ', len(df_model[(df_model['lang']=='en') & (df_model['label']=='ham')]))

#running tensorflow at backend
import keras
from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard

import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
#Create tf-idf model from the data
# max number of features
num_max = 4000
def train_tf_idf_model(texts):
    "train tf idf model "
    tic = time.process_time()
    

    tok = Tokenizer(num_words=num_max)
    tok.fit_on_texts(texts)
    toc = time.process_time()

    print (" -----total Computation time = " + str((toc - tic)) + " seconds")
    return tok


def prepare_model_input(tfidf_model,dataframe,mode='tfidf'):
    
    "function to prepare data input features using tfidf model"
    tic = time.process_time()
    
    le = LabelEncoder()
    sample_texts = list(dataframe['tokenized_text'])
    sample_texts = [' '.join(x.split()) for x in sample_texts]
    
    targets=list(dataframe['label'])
    targets = [1. if x=='spam' else 0. for x in targets]
    sample_target = le.fit_transform(targets)
    
    if mode=='tfidf':
        sample_texts=tfidf_model.texts_to_matrix(sample_texts,mode='tfidf')
    else:
        sample_texts=tfidf_model.texts_to_matrix(sample_texts)
    
    toc = time.process_time()
    
    print('shape of labels: ', sample_target.shape)
    print('shape of data: ', sample_texts.shape)
    
    print (" -----total Computation time for preparing model data = " + str((toc - tic)) + " seconds")
    
    return sample_texts,sample_target

#Split Train/validation data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(mat_texts, tags, test_size=0.15)
print ('train data shape: ', X_train.shape, y_train.shape)
print ('validation data shape :' , X_val.shape, y_val.shape)

#deep neural network model
## Define and initialize the network

model_save_path="checkpoints/spam_detector_enron_model.h5"
def get_simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc',keras.metrics.binary_accuracy])
    print('compile done')
    return model

def check_model(model,x,y,epochs=2):
    history=model.fit(x,y,batch_size=32,epochs=epochs,verbose=1,shuffle=True,validation_split=0.2,
              callbacks=[checkpointer, tensorboard]).history
    return history


def check_model2(model,x_train,y_train,x_val,y_val,epochs=10):
    history=model.fit(x_train,y_train,batch_size=64,
                      epochs=epochs,verbose=1,
                      shuffle=True,
                      validation_data=(x_val, y_val),
                      callbacks=[checkpointer, tensorboard]).history
    return history

# define checkpointer
checkpointer = ModelCheckpoint(filepath=model_save_path,
                               verbose=1,
                               save_best_only=True)    


# define tensorboard
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)




# define the predict function for the deep learning model for later use
def predict(data):
    result=spam_model_dl.predict(data)
    prediction = [round(x[0]) for x in result]
    return prediction

# get the compiled model
model = get_simple_model()

# load history
# history=check_model(m,mat_texts,tags,epochs=10)
history=check_model2(model,X_train,y_train,X_val,y_val,epochs=10)

#SVM

spam_model_svm = svm.SVC(verbose=1)
spam_model_svm.fit(X_train,y_train)

#random forest
from sklearn.ensemble import RandomForestClassifier
spam_model_rf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=50)

#train xgboost model
# Build xgboost also 
import xgboost as xgb
spam_model_xgboost = xgb.XGBClassifier()
spam_model_xgboost.fit(X_train,y_train)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
spam_model_rf.fit(X_train,y_train)

#prepare test data
sample_texts,sample_target=prepare_model_input(tfidf_model,df_unseen_test,mode='')
sample_texts,sample_target=prepare_model_input(tfidf_model,df_unseen_test,mode='')

#Result
result_df,results_cm= getResults(model_dict,sample_texts,sample_target)
result_df

#Plot confusion Matrix for all the models
def plot_heatmap(cm,title):
    df_cm2 = pd.DataFrame(cm, index = ['normal', 'spam'])
    df_cm2.columns=['normal','spam']

    ax = plt.axes()
    sns.heatmap(df_cm2, annot=True, fmt="d", linewidths=.5,ax=ax)
    ax.set_title(title)
    plt.show()

    return

#deep neural network
plot_heatmap(results_cm['deep_learning'],'Deep Learning')

#Xgboost
plot_heatmap(results_cm['xgboost'],'xgboost')

#for SVM
plot_heatmap(results_cm['svm'],'SVM')

#for random forest
plot_heatmap(results_cm['random_forest'],'Random Forest')
