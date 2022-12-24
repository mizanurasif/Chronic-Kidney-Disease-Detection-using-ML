from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import pickle
import pandas as pd
import numpy as np




df = pd.read_csv("kidney_Disease_Pre_processed.csv")
df.head(20)

x = df.drop(['Unnamed: 0','class'],axis='columns')
y = df['class']
x = x.values
y = y.values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#for i in df_imputed.columns:
    #print("***************************",i,"*************************************")
    #print()
    #print(set(df_imputed[i].tolist()))
    #print()

svm = SVC(C= 100, gamma= 'scale', kernel= 'linear')
knn = KNeighborsClassifier(algorithm = 'ball_tree', n_jobs= 1, n_neighbors= 6, weights= 'uniform')
lr= LogisticRegression(C = 100, max_iter = 2000, penalty= 'l2', solver = 'newton-cg')
xgb = XGBClassifier(colsample_bytree = 0.5, gamma= 0.0, learning_rate = 0.15, max_depth = 5, min_child_weight = 1)
rf = RandomForestClassifier(bootstrap =True, criterion = 'entropy', max_features = 'auto', min_samples_leaf = 1, min_samples_split = 4, n_estimators = 100)
adab = AdaBoostClassifier(base_estimator=GaussianNB(),learning_rate = 0.2, n_estimators = 10)


svm.fit(x_train,y_train)
print("train Accuracy SVM: %0.3f" % svm.score(x_train,y_train))
print("Test Accuracy SVM: %0.3f" % svm.score(x_test,y_test))

knn.fit(x_train,y_train)
print("train Accuracy KNN: %0.3f" % knn.score(x_train,y_train))
print("Test Accuracy KNN: %0.3f" % knn.score(x_test,y_test))

lr.fit(x_train,y_train)
print("train Accuracy LR: %0.3f" % lr.score(x_train,y_train))
print("Test Accuracy LR: %0.3f" % lr.score(x_test,y_test))

xgb.fit(x_train,y_train)
print("train Accuracy XGB: %0.3f" % xgb.score(x_train,y_train))
print("Test Accuracy XGB: %0.3f" % xgb.score(x_test,y_test))

rf.fit(x_train,y_train)
print("train Accuracy RF: %0.3f" % rf.score(x_train,y_train))
print("Test Accuracy  RF: %0.3f" % rf.score(x_test,y_test))

adab.fit(x_train,y_train)
print("train Accuracy ADAB: %0.3f" % adab.score(x_train,y_train))
print("Test Accuracy ADAB: %0.3f" % adab.score(x_test,y_test))





pickle.dump(svm,open('svm.pkl','wb'))
pickle.dump(knn,open('knn.pkl','wb'))
pickle.dump(lr,open('lr.pkl','wb'))
pickle.dump(xgb,open('xgb.pkl','wb'))
pickle.dump(rf,open('rf.pkl','wb'))
pickle.dump(adab,open('abab.pkl','wb'))
