#Importing Libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder as OHE
from datetime import date
import numpy as np
import pandas as pd
from langdetect import detect
from nltk import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

lst = [2 ,3 , 4, 6,7]
#Reading the train and test data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train = df_train.iloc[:,0:9]

non_integer_train = df_train.iloc[: , lst]
non_integer_test = df_test.iloc[: ,[2,3,4,6]]

non_integer_train.fillna('Z' , inplace = True)
non_integer_test.fillna('Z' , inplace = True)

#Encoding the classes
LB = LabelEncoder()

non_integer_train = non_integer_train.apply(LB.fit_transform)
non_integer_test = non_integer_test.apply(LB.fit_transform)


#imputing the last two coloumns

Imputer_response = SimpleImputer(missing_values = 10 , strategy = 'mean')
Imputer_dispute = SimpleImputer(missing_values = 2 , strategy = 'mean')

non_integer_train.iloc[: ,2:4] = Imputer_response.fit_transform(non_integer_train.iloc[: ,2:4])
non_integer_train.iloc[: ,4:5] = Imputer_dispute.fit_transform(non_integer_train.iloc[: ,4:5])

non_integer_test.iloc[:,2:3 ] = Imputer_response.transform(non_integer_test.iloc[: ,2:3])
non_integer_test.iloc[: ,3:4] =  Imputer_dispute.transform(non_integer_test.iloc[: ,3:4])

#Finding the time taken to reach the company
days_train = []
days_test = []

for i in range(len(df_train['Date-received'])):
    sent_data = df_train['Date-sent-to-company'][i]
    sent_data = sent_data.split('/')
    sent = date(int(sent_data[2]),int(sent_data[0]), int (sent_data[1]))
    
    received_data = df_train['Date-received'][i]
    received_data = received_data.split('/')
    received = date(int(received_data[2]),int(received_data[0]), int(received_data[1]))
    
    days_train.append((sent-received).days)

for i in range (len(df_test['Date-received'])):
    sent_data = df_test['Date-sent-to-company'][i]
    sent_data = sent_data.split('/')
    sent = date(int(sent_data[2]),int(sent_data[0]), int (sent_data[1]))
    
    received_data = df_test['Date-received'][i]
    received_data = received_data.split('/')
    received = date(int(received_data[2]),int(received_data[0]), int(received_data[1]))
    
    days_test.append((sent-received).days)

#Splitting train data into Eng, Fr and Es
train_summary_en =[]
train_summary_fr =[]
train_summary_es =[]
pos_train_en = []
pos_train_fr = []
pos_train_es = []

for i in range(len(df_train['Complaint-ID'])):
    summary = df_train['Consumer-complaint-summary'][i]
    if detect(summary)== 'en':
        #train_summary_en.append(summary)
        pos_train_en.append(i)
    elif detect(summary)== 'fr':
        #train_summary_fr.append(summary)
        pos_train_fr.append(i)
    else:
        #train_summary_es.append(summary)
        pos_train_es.append(i)
    print(i , ' done...')

#Splitting test data into Eng,Fr and Es
test_summary_en =[]
test_summary_fr =[]
test_summary_es =[]

pos_test_en = []
pos_test_fr = []
pos_test_es = []
    
from nltk.stem.wordnet import WordNetLemmatizer

for i in range(len(df_test['Complaint-ID'])):
    summary = df_test['Consumer-complaint-summary'][i]
    if detect(summary)== 'en':
        test_summary_en.append(summary)
        pos_test_en.append(i)
    elif detect(summary)== 'fr':
        test_summary_fr.append(summary)
        pos_test_fr.append(i)
    else:
        test_summary_es.append(summary)
        pos_test_es.append(i)
    print(i , ' done..')

#Cleaning the summary
count = 0
def clean_summary(text_data, lang):
    global count
    clnr = re.compile('<.*?>')
    text_data = re.sub(clnr, ' ' ,text_data)
    text_data = re.sub('[^a-zA-Z]' , '  ' ,text_data)
    ps = PorterStemmer()
    text_data = text_data.lower()
    
    text_data = nltk.word_tokenize(text_data)
    
    lmt = WordNetLemmatizer()

    text_data = [ ps.stem(i) for i in text_data if not i in set(stopwords.words(lang))]
    text_data = [lmt.lemmatize(i) for i in text_data]
    text_data = ' '.join(text_data)
    text_data = text_data.lower()
    count = count +1 
    print(count ,' done..')
    return(text_data)

traincpy_summary_en = [clean_summary(i,'english') for i in train_summary_en]

testcpy_summary_en = [clean_summary(i,'english') for i in test_summary_en]

traincpy_summary_fr = [clean_summary(i,'french') for i in train_summary_fr]

testcpy_summary_fr = [clean_summary(i, 'french') for i in test_summary_fr]

traincpy_summary_es = [clean_summary(i, 'spanish') for i in train_summary_es]

testcpy_summary_es = [clean_summary(i , 'spanish') for i in test_summary_es]

lensummary_es = [ len(i) for i in traincpy_summary_es ]  
lensummary_fr = [ len(i) for i in traincpy_summary_fr ]
lensummary_en = [ len(i) for i in traincpy_summary_en ]

#Splitting the dataset based on the language i.e en, es or fr
df_train_en = pd.DataFrame()
df_train_fr = pd.DataFrame()
df_train_es = pd.DataFrame()

df_test_en = pd.DataFrame()
df_test_fr = pd.DataFrame()
df_test_es = pd.DataFrame()

df_train_en = non_integer_train.iloc[pos_train_en, [0,1,2,4]]
df_train_en['Days'] = [days_train[i] for i in pos_train_en ]
df_train_en['Summary'] = traincpy_summary_en
df_train_en['output'] = non_integer_train.iloc[pos_train_en, 3]

df_train_fr = non_integer_train.iloc[pos_train_fr, :]
df_train_fr['Days'] = [days_train[i] for i in pos_train_fr ]
df_train_fr['Summary'] = traincpy_summary_fr


df_train_es = non_integer_train.iloc[pos_train_es, :]
df_train_es['Days'] = [days_train[i] for i in pos_train_es ]
df_train_es['Summary'] = traincpy_summary_es

df_test_en = non_integer_test.iloc[pos_test_en, :]
df_test_en['Days'] = [days_test[i] for i in pos_test_en ]
df_test_en['Summary'] = testcpy_summary_en
df_test_en['pos'] = pos_test_en

df_test_fr = non_integer_test.iloc[pos_test_fr, :]
df_test_fr['Days'] = [days_test[i] for i in pos_test_fr ]
df_test_fr['Summary'] = testcpy_summary_fr
df_test_fr['pos'] = pos_test_fr

df_test_es = non_integer_test.iloc[pos_test_es, :]
df_test_es['Days'] = [days_test[i] for i in pos_test_es ]
df_test_es['Summary'] = testcpy_summary_es
df_test_es['pos'] = pos_test_es

df_train_en.to_csv('train_en.csv')
df_train_fr.to_csv('train_fr.csv')
df_train_es.to_csv('train_es.csv')
df_test_en.to_csv('test_en.csv')
df_test_fr.to_csv('test_fr.csv')
df_test_es.to_csv('test_es.csv')
