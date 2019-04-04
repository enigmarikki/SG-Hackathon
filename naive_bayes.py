import pandas as pd
import numpy as np
from sklearn import model_selection 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

classify('train_en.csv' , 'test_en.csv' )

classify('train_fr.csv' , 'test_fr.csv' )

classify('train_es.csv' , 'test_es.csv' )

#Classifiying alg Naive_bayes
def classify(dataset_train , dataset_test):
    
    training = pd.read_csv(dataset_train)
    test = pd.read_csv(dataset_test)
    #change
    #Vectorizer
    print('vectorize started..')
    training['Summary'].fillna('A' , inplace = True)
    Vectorizer = TfidfVectorizer()
    print('vectorizer fitting started')
    Vectorizer.fit(training['Summary'])
    vector_train = Vectorizer.transform(training['Summary'])
    vector_test = Vectorizer.transform(test['Summary'])
    print(np.shape(vector_train))
    print('vectorize done..')
    X = training.iloc[: , [1,2,3,5,6]].values
    Y = training.iloc[:, 4].values
    X = np.concatenate((X , vector_train.toarray()), axis =1)
    Y = Y.astype('int')
    _X =  test.iloc[: , 1:6]
    _X = np.concatenate((_X,vector_test.toarray()) , axis = 1)
    #Uncomment this to use it as private train-test
    #X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.05)

    print('StartedFeatureScaling')

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    _X =sc.transform(_X) #use Xen_train instead
    #X_test = sc.fit_transform(X_test)
    #X_en_test = sc.fit_transform(X_en_test)

    print('fitting classifier')
    classifierNB = MultinomialNB()
    #Use Xen_train,Yen_train
    classifierNB.fit(X, Y)

    #Uncomment this to check for accuracy
    '''Y_pred = classifierNB.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(Y_test , Y_pred)
    print(cm1)'''
    #fit the test
    
    Yout = classifierNB.predict(_X)

    df= pd.DataFrame()
    #df['pos'] = posen
    df['output']= Yout
    df.to_csv('output'+ dataset_test)
