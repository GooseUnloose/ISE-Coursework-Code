#!!!IMPORTANT INFO BEFORE RUNNING!!!
#Should you execute this program it will train the loaded model across 5 datasets, where results of each test
#will output two CSV files, one containing the mean average reults accross 30 tests, and another the raw results
#of said tests. Suffice to say, this is a lot of files, please ensure the target directory you configure in
#logTest() wont become too cluttered at the introduction of all these test files.

## imports random and establishes a seed for this randomisation
import random

import time
import pandas as pd
import numpy as np
import re
import math
import statistics

#nltk import plus specifics
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#All Classification models imported for this tool

from sklearn.naive_bayes import GaussianNB,MultinomialNB

#Grid search, vectoriser and performance metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
#To permit dataset visualisations to be called in functions. This was used at some point during development, but may not be present in final submission
#Not wholy neccessary to the tool, just used for visualisation purposes during data cleaning. 
from IPython.display import display_html


random.seed(1234)
##generate list of random seeds for model training 
def genSeeds(inputList):
    for i in range(0,30):
            inputList.append(random.randint(0,9999))
    
    return inputList


#preprocess cleaning of datasets, removing irreleveant & metadata columns
def dfReduction(inpCSV):
    #this function anticipates a particular formatting of fields, and abreviates them into three columns, Title, Body and Target which will be where classifications are stored
    #it could be extended using some logic to normalise any given dataset into our frame standard
    newDf = pd.DataFrame().assign(id=inpCSV['Unnamed: 0'],Title=inpCSV['Title'],Body=inpCSV['Body'],Target=inpCSV['class'])
    
    return newDf

#preprocess cleaning of datasets, removing irreleveant & metadata columns for 1st-Phase 
def phase1Reduction(inpCSV):
    newDf = pd.DataFrame().assign(id=inpCSV['Unnamed: 0'],Title=inpCSV['Title'],Body=inpCSV['Body'],Quality=inpCSV['Quality'])
    
    return newDf

#extra preprocessing methods

#removes any html specific characters within text
def removeHTML(inpCSV):
    #As datasets have been standardised we can reference the column directly
    htmlSub = re.compile(r'<.*?>')
    inpCSV = htmlSub.sub(r'',inpCSV)
    return inpCSV

#removes stop words from dataset
nltk.download('stopwords')
def removeStopwords(inpCSV):
    stopwordsList = stopwords.words('english')
    for i in inpCSV.split():
        if i in stopwordsList:
            i = " "
    return inpCSV
    
def removeEmotes(inpCSV):
    emoteSub = re.compile("["
                               u"\U0001F600-\U0001F64F"  # removes emote characters
                               u"\U0001F300-\U0001F5FF"  # removes symbols
                               u"\U0001F680-\U0001F6FF"  # removes geographic characters
                               u"\U0001F1E0-\U0001F1FF"  # removes flags 
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # removes enclosed characters
                               "]+", flags=re.UNICODE)
    inpCSV = emoteSub.sub(r'',inpCSV)
    return inpCSV

#removes all punctuation and non alpha charcters from a text
def bodyCleaner(inpCSV):  
    inpCSV = re.sub("[^a-zA-Z ]+","",inpCSV)  
    return inpCSV.strip().lower()

#reduces all words to core or root forms
def Stemming(inpCSV):
    for i in inpCSV:
        PorterStemmer().stem(i)
    return inpCSV


def subsetCSV(inpCSV):
    #creates a subset of the input CSV containing only 60% of the original records, this will be the training set for each dataset
    #meaning for each dataset, the 1st phase model 40% of the data it encounters will be unseen
    subRecords = round(len(inpCSV) * 0.4)
    
    for i in range (0,subRecords):
        #select random records from DF and drop them to reduce the dataframe
        randRecord = random.randint(0,(len(inpCSV) - 1))
        inpCSV = inpCSV.drop(i)
    
    #return reduced training dataset for phase 1, this is to prevent model overfitting
    return inpCSV

#traning function for the first phase of the 2 phase solution
def trainPhase1(trainCSV):
    
    trainCSV = qualityAssess(trainCSV)
    trainCSV = phase1Reduction(trainCSV)
    trainCSV = subsetCSV(trainCSV)
    trainCSV['Body'] = trainCSV.apply(lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],axis=1)
    trainCSV['Body'] = trainCSV['Body'].apply(removeStopwords)
    trainCSV['Body'] = trainCSV['Body'].apply(removeHTML)
    trainCSV['Body'] = trainCSV['Body'].apply(bodyCleaner)
    params = {
        'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]
        }
    clf1Phase = MultinomialNB()
    grid = GridSearchCV(clf1Phase,params,cv = 5,scoring='roc_auc')
    tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=1000)
    trainText = trainCSV['Body']
    y_train = trainCSV['Quality']
    X_train = tfidf.fit_transform(trainText).toarray()
    grid.fit(X_train,y_train)
    bestClf = grid.best_estimator_
    bestClf.fit(X_train,y_train)
    return bestClf  

def runPhase1(inpCSV,model):
    #predicts utility of bug reports, then returns a subset of the original dataset where all elements are classified as useful

    #if there is no trained 1 phase model, the function returns an unmodified DF
    if model == None:
        return inpCSV
    else:
        originalCSV = inpCSV.copy()
        print(originalCSV.shape)
        inpCSV = inpCSV.drop(['Target'],axis='columns')
    
        tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=1000)
        X_train = tfidf.fit_transform(inpCSV['Body']).toarray()
        inpPred = model.predict(X_train)
        for i in range (0,len(inpPred)):
            if inpPred[i] == 0:
                originalCSV= originalCSV.drop(i)
        print(originalCSV.shape)
        return originalCSV

#main function for training the performance classification models
def train(id,phase,inpCSV,seeds,param,model):
    
    accuracies  = []
    precisions  = []
    recalls     = []
    f1_scores   = []
    aucValues  = []
    processTimes = []
       
    params = param
    for i in range(0,len(seeds)):
        
        startTime = time.time()
        
        indices = np.arange(inpCSV.shape[0])
        trainIndex, testIndex = train_test_split(indices,test_size=0.2,random_state=seeds[i])
        trainText = inpCSV['Body'].iloc[trainIndex]
        testText = inpCSV['Body'].iloc[testIndex]
        
        y_train = inpCSV['Target'].iloc[trainIndex]
        y_test = inpCSV['Target'].iloc[testIndex]
        
        tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=1000)
        
        X_train = tfidf.fit_transform(trainText).toarray()
        X_test = tfidf.transform(testText).toarray()
        
        clf = model
        clfName = type(clf).__name__
        grid = GridSearchCV(clf,params,cv = 5,scoring='roc_auc')
        
        grid.fit(X_train,y_train)
        
        bestClf = grid.best_estimator_
        bestClf.fit(X_train,y_train)
        
        y_pred = bestClf.predict(X_test)
        
        endTime = time.time() - startTime
        processTimes.append((endTime))
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        falsePR, truePR, Void = roc_curve(y_test,y_pred,pos_label=1)
        aucValues.append(auc(falsePR,truePR))
    
    print(f"Mean Accuracy:{np.mean(accuracies)}")
    print(f"Mean Precision:{np.mean(precisions)}")    
    print(f"Mean Recall:{np.mean(recalls)}")    
    print(f"Mean F1_Score:{np.mean(f1_scores)}")
    print(f"Mean AUC:{np.mean(aucValues)}")
    print(f"Mean Process Time:{np.mean(processTimes)}")
    
    logTest(id,clfName,check2Phase(phase),len(seeds),accuracies,precisions,recalls,f1_scores,aucValues,processTimes)        

#small function for checking if the test is 2 phase or not
def check2Phase(inp):
    if inp != None:
        return True
    
    else:
        return False

#writes results of tests to an external CSV file, one for mean values, the other for Raw values for each test seed. Naming the file appropriately 
def logTest(id,model,phase,freq,accuracy,precision,recall,f1_score,aucValues,processTime):
    
    if phase == False:
        out_csv_name = f'Results/1Phase_{id}_{model}'
        
    else:
        out_csv_name = f'Results/2Phase_{id}_{model}'
        
    try:
    # Attempt to check if the file already has a header
        xisting_data = pd.read_csv(out_csv_name, nrows=1)
        header_needed = False
    except:
        header_needed = True
    
    df_log = pd.DataFrame(
    {
        'repeated_times': [freq],
        'Accuracy': [np.mean(accuracy)],
        'Precision': [np.mean(precision)],
        'Recall': [np.mean(recall)],
        'F1': [np.mean(f1_score)],
        'AUC': [np.mean(aucValues)],
        'Process Time' : [np.mean(processTime)]
    })
    df_log.to_csv((out_csv_name + ".csv"), mode='a', header=header_needed, index=False)
    df_log = pd.DataFrame(
    {
        'repeated_times': freq,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score,
        'AUC': aucValues,
        'Process Time' : processTime
    })
    df_log.to_csv((out_csv_name + "_RAWDATA.csv"), mode='a', header=header_needed, index=False)


##Qualiative classification function
def qualityAssess(inpCSV):
    #retireve mean length of bug report language, all reports 2 SD out of this mean are classed as irrelivant
    bodyLength = []
    relevanceTag = []
    for i in inpCSV['Body']:
        bodyLength.append(len(str(i).split()))
    print(f"Mean body length: {statistics.mean(bodyLength)}")
    print(f"Standard deviation of body length {statistics.stdev(bodyLength)}")
    print(f"Max length found: {max(bodyLength)}\nMin length found:{min(bodyLength)}\n")
    for i in bodyLength:
        if i < 10:
            relevanceTag.append(0)
        else:
            if (i > (statistics.mean(bodyLength) + statistics.stdev(bodyLength)) or (i < (statistics.mean(bodyLength) - statistics.stdev(bodyLength)))):  
                relevanceTag.append(0)
            else:
                relevanceTag.append(1)
    
    inpCSV['Quality'] = relevanceTag
    print(inpCSV['Quality'].value_counts())
    return inpCSV

def dfCycle():
    #Runs functionality across all datasets, useful for automative testing
    #All processes for cleaning, testing and recording results are done in this function
    models = [GaussianNB(),MultinomialNB()]
    parameters = [{'var_smoothing': np.logspace(-12, 0, 13)},{'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]}]
    for j in range(0,len(models)):
        print(f"\nactive Model: {type(models[j]).__name__}")
        datasets={'caffe','incubator-mxnet','keras','pytorch','tensorflow'}
        for i in datasets:
            print(f"\n\n{i}\n")
            df = pd.read_csv(f"Datasets/{i}.csv").fillna("")
            df = dfReduction(df)
            df['Body'] = df.apply(lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],axis=1)
            df = df.drop('Title',axis='columns')
            df['Body'] = df['Body'].apply(removeEmotes)
            df['Body'] = df['Body'].apply(removeStopwords)
            df['Body'] = df['Body'].apply(removeHTML)
            df['Body'] = df['Body'].apply(bodyCleaner)
        

            df1 = pd.read_csv(f"Datasets/{i}.csv").fillna("")

            #determins if the multinomial model is being run currently, will then perform both 1 and 2 phase testing
            if type(models[j]).__name__ == "MultinomialNB":
                for k in range(0,2):
                    if k == 0:
                        print("2-Phase test")
                        p1 = trainPhase1(df1)
                        df = runPhase1(df,p1)
                        df['Body'] = df['Body'].apply(Stemming)
                        train(i,p1,df,trainSeeds,parameters[j],models[j])
                    else:
                        p1 = None
                        df = runPhase1(df,p1)
                        df['Body'] = df['Body'].apply(Stemming)
                        train(i,p1,df,trainSeeds,parameters[j],models[j])
            else:
                ##To perform standard, non 2-phase approach make p1 = None, the trainPhase1 function has a condition for handling none values
                print("1-Phase test")
                p1 = None
                df = runPhase1(df,p1)
                df['Body'] = df['Body'].apply(Stemming)
                train(i,p1,df,trainSeeds,parameters[j],models[j])

    print("\nDone! ^-^")
        

##generates 30 random seeds to trainSeeds array
trainSeeds = []
genSeeds(trainSeeds)

dfCycle()
