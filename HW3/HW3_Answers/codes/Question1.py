###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import standardize
from sklearn.metrics import f1_score,accuracy_score

if __name__=='__main__':
    data = pd.read_excel("LSVT_voice_rehabilitation.xlsx", sheet_name=0)
    data = standardize(data)
    labels = pd.read_excel("LSVT_voice_rehabilitation.xlsx", sheet_name=1)
    data['class'] = labels
    X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:, data.columns != 'class'],
                                                        data['class'], test_size=0.25, random_state=42)
    # linear kernel
    # svclassifier = SVC(kernel='linear',C=0.001)

    # polynomial Kernel
    svclassifier = SVC(kernel='poly',degree=3,C=1000,max_iter=1e5)

    # RBF Kernel
    # svclassifier = SVC(kernel='rbf',gamma='auto')

    # Sigmoid Kernel
    # svclassifier = SVC(kernel='sigmoid', C=2.5)

    svclassifier.fit(X_train, Y_train)
    y_pred = svclassifier.predict(X_test)
    print("Accuracy : ",accuracy_score(Y_test,y_pred))
    print("F1_score : ",f1_score(Y_test, y_pred))


