###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import itertools
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import math
from random import randrange
import random
from sklearn.metrics import roc_curve, roc_auc_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# gaussClf will be the class that will have the Gaussian naive bayes classifier implementation
class gaussClf:

    def separate_by_classes(self, X, y):
        ''' This function separates our dataset in subdatasets by classes '''
        self.classes = np.unique(y)
        classes_index = {}
        subdatasets = {}
        cls, counts = np.unique(y, return_counts=True)
        self.class_freq = dict(zip(cls, counts))
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(y==class_type)
            subdatasets[class_type] = X[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type]/sum(list(self.class_freq.values()))
        return subdatasets


    def fit(self, X, y):
        ''' The fitting function '''
        separated_X = self.separate_by_classes(X, y)
        self.means = {}
        self.std = {}
        for class_type in self.classes:
            # Here we calculate the mean and the standart deviation from datasets
            self.means[class_type] = np.mean(separated_X[class_type], axis=0)[0]
            self.std[class_type] = np.std(separated_X[class_type], axis=0)[0]


    def calculate_probability(self, x, mean, stdev):
        ''' This function calculates the class probability using gaussian distribution '''
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


    def predict_proba(self, X):
        ''' This function predicts the probability for every class '''
        self.class_prob = {cls: math.log(self.class_freq[cls], math.e) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(self.means)):
                # print(X[i])
                self.class_prob[cls] += math.log(self.calculate_probability(X[i], self.means[cls][i], self.std[cls][i]),
                                                 math.e)
        self.class_prob = {cls: math.e ** self.class_prob[cls] for cls in self.class_prob}
        return self.class_prob


    def predict(self, X):
        ''' This funtion predicts the class of a sample '''
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for cls, prob in self.predict_proba(x).items():
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred


# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


#splitting data into train and test datasets
def train_test_split(data,ratio):
    X = (data.loc[:,data.columns != 'class'])
    s = int(ratio * len(X))
    x_train = X[:s]
    x_test = X[s:]
    Y = (data['class'])
    s = int(ratio * len(Y))
    y_train = Y[:s]
    y_test = Y[s:]
    return x_train,y_train,x_test,y_test


#plot ROC curve
def ROC(y_test,y_prob):
    colors = itertools.cycle(['m', 'darkorange', 'darkcyan'])
    for i,color in zip(range(len(y_prob)),colors):
        fpr1, tpr1, thresh1 = roc_curve(y_test, y_prob[i], pos_label=i+1)
        roc_auc = metrics.auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, lw=2,color=color, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def part1():
    data = pd.read_csv('wine.data', header=None)
    data.columns = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols','Flavanoids',
                    'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                    'Proline']
    data = data.sample(frac=1).reset_index(drop=True)
    k = 6
    datasets = cross_validation_split(data.to_numpy(),k)
    accuracies = []
    for i in range(k):
        test_data = datasets[i]
        df_test= pd.DataFrame(test_data, columns=data.columns)
        y_test = df_test[['class']].to_numpy()
        x_test = df_test.loc[:, df_test.columns != 'class'].to_numpy()
        train_data = [d for d in datasets if d is not test_data]
        train_data = list(itertools.chain.from_iterable(train_data))
        df_train = pd.DataFrame(train_data,columns=data.columns)
        y_train = df_train[['class']].to_numpy()
        x_train = df_train.loc[:, df_train.columns != 'class'].to_numpy()
        GNB = gaussClf()
        GNB.fit(x_train,y_train)
        y_pred = GNB.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        accuracies.append(accuracy)
    print("accuracy is : {}".format(np.mean(accuracies)*100))


def part2():
    data = pd.read_csv('wine.data', header=None)
    data.columns = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids',
                    'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                    'Proline']
    data = data.sample(frac=1).reset_index(drop=True)
    x_train, y_train, x_test, y_test =train_test_split(data,0.7)
    GNB = gaussClf()
    GNB.fit(x_train.to_numpy(), y_train.to_numpy())
    y_pred = GNB.predict(x_test.to_numpy())
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy * 100)

    #plotting ROC curve
    y_prob=[[],[],[]]
    for i in range(3):
        for x in x_test.to_numpy():
            y = GNB.predict_proba(x).get(i+1)
            y_prob[i].append(y)
    ROC(y_test,y_prob)

if __name__=='__main__':
    part1()
    # part2()
