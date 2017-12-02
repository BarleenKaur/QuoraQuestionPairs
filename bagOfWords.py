from sklearn.feature_extraction.text import CountVectorizer
import codecs, csv, sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split 
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import time

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def vectorizer(x_train, x_test):
    # vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', stop_words="english", tokenizer=LemmaTokenizer())
    vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', stop_words="english", tokenizer=None)
    vectorizer.fit(x_train)                                         # X_train == our data dictionary
    x_train_matrix = vectorizer.transform(x_train)                  #transformoing testing data with the our defined dictionary
    x_test_matrix = vectorizer.transform(x_test)
    
    return x_train_matrix, x_test_matrix
def naive_bayer(x_train_matrix, x_test_matrix, y_train, y_test):    #NAIVE BAYERS MODEL
                                                                    
    nb = MultinomialNB(alpha=2.8)
    nb.fit(x_train_matrix, y_train)                                 #train the model
    y_pred = nb.predict(x_test_matrix)                              #make predictions for X_test
    print ("Naive Byers Model score: " + str(accuracy_score(y_test, y_pred)))         
    print ("Naive Byers Model confusion matrix:")
    print (confusion_matrix(y_test, y_pred))

def logistic_regression(x_train_matrix, x_test_matrix, y_train, y_test): 
    logReg = LogisticRegression(C=0.85)
    logReg.fit(x_train_matrix, y_train)
    y_pred_log = logReg.predict(x_test_matrix)
    print ("Logistic Regression score: "+ str(accuracy_score(y_test, y_pred_log)))
    print ("Logistic Regression confusion matrix:")
    print (confusion_matrix(y_test, y_pred_log))

def linear_svm(x_train_matrix, x_test_matrix, y_train, y_test):
    svm1 = LinearSVC(C=1)
    svm1.fit(x_train_matrix, y_train)
    y_pred_svc = svm1.predict(x_test_matrix)
    print ("SVC score: " + str(accuracy_score(y_test, y_pred_svc)))
    print ("SVC confusionmatrix:")
    print (confusion_matrix(y_test, y_pred_svc))


def main():
    start_time = time.time()

    train_questions = []
    train_label = []

    f = open('train.csv','r')
    reader = csv.DictReader(f)
    for row in reader:
        train_questions.append(row['question1'].lower() + row['question2'].lower())
        train_label.append( row['is_duplicate'])
    train_question_label = zip(train_questions,train_label) 
    train_df = pd.DataFrame(train_question_label, columns=['sentense', 'label'])
    X = train_df.sentense
    Y = train_df.label

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1)
    X_train_matrix, X_test_matrix = vectorizer(X_train, X_test)

    preprocessing_time = time.time()
    print("--- start -> preprocessing:%s seconds ---" % (preprocessing_time - start_time))

    naive_bayer(X_train_matrix, X_test_matrix, Y_train, Y_test)
    naive_bayer_time = time.time() 
    print("--- preprocessing -> naive bayer :%s seconds ---" % (naive_bayer_time - preprocessing_time))
    
    print ("-------------------------------------------")

    logistic_regression(X_train_matrix, X_test_matrix, Y_train, Y_test)
    logistic_regression_time = time.time()
    print("--- niave bayer -> logistic regression :%s seconds ---" % (logistic_regression_time - naive_bayer_time))

    print ("-------------------------------------------")

    linear_svm(X_train_matrix, X_test_matrix, Y_train, Y_test)
    linear_svm_time = time.time()
    print("--- logistic regression -> linear svm :%s seconds ---" % (linear_svm_time - logistic_regression_time))
    print("--- TOTAL TIME: %s seconds ---" % (time.time() - start_time))
    


if __name__ == '__main__':
    main()









# test_questions = []
# test_label = []

# f = open('test.csv','r')
# reader = csv.DictReader(f)
# for row in reader:
#     test_questions.append(row['question1'] + row['question2'])
#     test_label.append( row['is_duplicate'])

# test_question_label = zip(test_questions,test_label) 
# test_df = pd.DataFrame(test_question_label, columns=['sentense', 'label'])
# X_test = test_df.sentense
# Y_test = test_df.label
#bigram & stop