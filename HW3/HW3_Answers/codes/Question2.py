###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import pandas as pd
from sklearn.utils import shuffle
from mlxtend.preprocessing import standardize
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from  xgboost import XGBClassifier

def load_data(train_path,test_path):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    data_train = shuffle(data_train)
    data_train.columns = [str(i) for i in range(1, 18)]
    data_test.columns = data_train.columns
    X_train = data_train.iloc[:, 0:16]
    X_train = standardize(X_train)
    Y_train = data_train.iloc[:, 16]
    X_test = data_test.iloc[:, 0:16]
    X_test = standardize(X_test)
    Y_test = data_test.iloc[:, 16]
    return X_train,Y_train,X_test,Y_test

def RandomForest(X_train, Y_train,X_test,Y_test):
    cls = RandomForestClassifier(n_estimators=15,max_depth=3,max_features=3)
    cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test)
    print('accuracy of Random Forest: ', metrics.accuracy_score(Y_test, y_pred))
    # finding confusion_matrix
    print(confusion_matrix(Y_test, y_pred))

def AdaBoost(X_train, Y_train,X_test,Y_test):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=10)
    ada1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=5)
    ada2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=20)
    ada3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=50)
    classifiers = [ada, ada1, ada2, ada3]
    for cls in classifiers:
        cls.fit(X_train, Y_train)
        y_pred = cls.predict(X_test)
        print(f'accuracy of adaboost with {cls.n_estimators} estimators:', metrics.accuracy_score(Y_test, y_pred))

def GradBoosting(X_train, Y_train,X_test,Y_test):
    model = XGBClassifier(learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     reg_lambda=0.1,
     objective= 'logistic',
     nthread=-1,
     scale_pos_weight=1,
     booster='gbtree',
     seed=77)
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    print('accuracy of xgb :', metrics.accuracy_score(Y_test, y_pred))

if __name__=='__main__':
    X_train,Y_train,X_test,Y_test= load_data("data_train.csv","data_test.csv")
    RandomForest(X_train,Y_train,X_test,Y_test)
    AdaBoost(X_train, Y_train,X_test,Y_test)
    GradBoosting(X_train, Y_train,X_test,Y_test)




