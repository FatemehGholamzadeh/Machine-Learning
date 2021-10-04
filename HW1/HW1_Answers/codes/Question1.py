###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import numpy as np
import pandas as pd
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('ggplot')

#class PolynomialRegression for generating objects for linear regression
class PolynomialRegression(object):

    def __init__(self, x_train, y_train,x_test, y_test):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def standardize(self,data):
        return (data - np.mean(data))/(np.max(data) - np.min(data))
        
    def hypothesis(self, theta, x):
        h = theta[0]
        for i in np.arange(1, len(theta)):
            h += theta[i]*x ** i        
        return h
        
    def compute_MSE(self, x, y, theta):
        m = len(y)  
        h = self.hypothesis(theta, x)
        errors = h-y
        return (1/(m))*np.sum(errors**2)

    def compute_RMSE(self, x, y, theta):
        m = len(y)
        h = self.hypothesis(theta, x)
        errors = h-y
        return np.math.sqrt((1 / (m)) * np.sum(errors ** 2))

    def compute_MAE(self, x, y, theta):
        m = len(y)
        h = self.hypothesis(theta, x)
        errors = h-y
        return np.mean(abs(errors))
        
    def fit(self, method = 'normal_equation', order = 1, tol = 10**-3, numIters = 20, learningRate = 0.01, error = 'MSE' , l=5):

        self.order = order
        # normal equation without lambda
        if method == 'normal_equation':
            d = {}
            d['x' + str(0)] = np.ones([1, len(self.x_train)])[0]
            for i in np.arange(1, order+1):
                d['x' + str(i)] = self.x_train ** (i)
            d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
            X = np.column_stack(d.values())
            theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X),X)), np.transpose(X)), self.y_train)
            MSE = self.compute_MSE(self.x_test, self.y_test, theta)
            RMSE = self.compute_RMSE(self.x_test, self.y_test, theta)
            MAE = self.compute_MAE(self.x_test, self.y_test, theta)
            print("Normal Equation, Order: "+str(self.order))
            print("MSE: "+str(MSE))
            print("RMSE: "+str(RMSE))
            print("MAE: "+str(MAE))
            print("************************************")

        # gradient_descent
        if method == 'gradient_descent':
            d = {}
            d['x' + str(0)] = np.ones([1,len(self.x_train)])[0]
            for i in np.arange(1, order+1):
                d['x' + str(i)] = self.standardize(self.x_train ** (i))
            d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
            X = np.column_stack(d.values())
            m = len(self.x_train)
            theta = np.zeros(order + 1)
            costs_train = []
            costs_test = []
            paces = []
            for i in range(numIters):
                h = self.hypothesis(theta, self.x_train)
                errors = h-self.y_train
                pace = learningRate * (1 / m) * np.dot(errors, X)
                paces.append(pace)
                theta += -pace
                if(error == 'MSE'):
                    cost_train = self.compute_MSE(self.x_train, self.y_train, theta)
                    cost_test = self.compute_MSE(self.x_test, self.y_test, theta)
                elif (error == 'RMSE'):
                    cost_train = self.compute_RMSE(self.x_train, self.y_train, theta)
                    cost_test = self.compute_RMSE(self.x_test, self.y_test, theta)
                else:
                    cost_train = self.compute_MAE(self.x_train, self.y_train, theta)
                    cost_test = self.compute_MAE(self.x_test, self.y_test, theta)
                costs_train.append(cost_train)
                costs_test.append(cost_test)
                #tolerance check
                if cost_train < tol:
                    break
            self.costs_train = costs_train
            self.costs_test = costs_test
            self.numIters = numIters
            self.error=error
            self.paces= paces
            print("order: "+str(self.order))
            print("iterations number: " + str(self.numIters ))
            print(self.error + " for Train:")
            print(self.costs_train.__getitem__(len(self.costs_train) - 1))
            print(self.error + " for Test:")
            print(self.costs_test.__getitem__(len(self.costs_test)-1))
            print("**************************************************")

        #normal equation with lambda
        if method == 'normal_equation_2':
            d = {}
            d['x' + str(0)] = np.ones([1, len(self.x_train)])[0]
            for i in np.arange(1, order+1):
                d['x' + str(i)] = self.x_train ** (i)
            d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
            X = np.column_stack(d.values())
            identity = np.identity(order+1)
            identity[0][0] = 0
            theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X),X) + l * identity ), np.transpose(X)), self.y_train)
            print(theta)
            MSE = self.compute_MSE(self.x_test, self.y_test, theta)
            MSE_train = self.compute_MSE(self.x_train, self.y_train, theta)
            print("Normal Equation, Order: "+str(self.order))
            print("MSE Test: "+str(MSE))
            print("MSE Train: "+str(MSE_train))
            print("************************************")
            self.l = l

        self.theta = theta
        self.method = method
        return self

     #plotting fitted curve
    def plot_predictedPolyLine(self,axs,j,k):
        axs[j][k].scatter(self.x_train, self.y_train, s = 30, c = 'c')
        axs[j][k].scatter(self.x_test, self.y_test, s = 30, c = 'orangered')
        line = self.theta[0] #y-intercept
        label_holder = []
        label_holder.append('%.*f' % (2, self.theta[0]))
        for i in np.arange(1, len(self.theta)):            
            line += self.theta[i] * self.x_test ** i
            label_holder.append(' + ' +'%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$')
        axs[j][k].plot(self.x_test, line,"*", label = ''.join(label_holder),color='indigo')
        axs[j][k].set_title('Order: '+str(len(self.theta)-1) +', Iterations#='+str(self.numIters)+', error='+self.error)
        axs[j][k].set_xlabel('x')
        axs[j][k].set_ylabel('y')
        axs[j][k].legend(loc = 'best')
        # plt.show()

    # plotting fitted curve for normal equation
    def plot_predictedPolyLine_normalEq(self,axs,j):
        axs[j].scatter(self.x_train, self.y_train, s = 30, c = 'c')
        axs[j].scatter(self.x_test, self.y_test, s = 30, c = 'orangered')
        line = self.theta[0] #y-intercept
        label_holder = []
        label_holder.append('%.*f' % (2, self.theta[0]))
        for i in np.arange(1, len(self.theta)):
            line += self.theta[i] * self.x_test ** i
            label_holder.append(' + ' +'%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$')
        axs[j].plot(self.x_test, line,"*", label = ''.join(label_holder),color='indigo')
        axs[j].set_title('Normal Equation, Order= '+str(len(self.theta)-1))
        # axs[j].set_title('Normal Equation, Order= '+str(len(self.theta)-1)+", lambda= "+str(self.l))
        axs[j].set_xlabel('x')
        axs[j].set_ylabel('y')
        axs[j].legend(loc = 'best')

    # plotting cost of Test and Train
    def plotCost(self,axs,j,k):
        if self.method == 'gradient_descent':
            # plt.figure()
            axs[j][k].plot(np.arange(1, self.numIters+1), self.costs_train,color="darkcyan", label = "cost of train")
            axs[j][k].plot(np.arange(1, self.numIters+1), self.costs_test,color="orangered",label = "cost of test")
            axs[j][k].set_xlabel('Iterations')
            axs[j][k].set_ylabel(r'$J(\theta)$')
            axs[j][k].set_title('Cost vs Iterations: order='+str(self.order)+' ,Itrs#='+str(self.numIters)+', error='+self.error)
            axs[j][k].legend(loc = 'best')
            # plt.show()
        else:
            print('plotCost method can only be called when using gradient descent method')

    # plotting pace size vs iterations
    def plotPace(self,axs,j,k):
        if self.method == 'gradient_descent':
            # plt.figure()
            paces = []
            for i in range(len(self.paces)):
                paces.append(self.paces[i][1])
            axs[j][k].plot(np.arange(1, self.numIters+1), paces,color="m")
            axs[j][k].set_xlabel('Iterations')
            axs[j][k].set_ylabel('pace size')
            axs[j][k].set_title('pace size vs iteraions order='+str(self.order)+' ,Itrs#='+str(self.numIters)+', error='+self.error)
            axs[j][k].legend(loc = 'best')
            # plt.show()
        else:
            print('plotPace method can only be called when using gradient descent method')

#normalizing our data
def normalizer(dataFrame):
    maxX = dataFrame['x'].max()
    minX = dataFrame['x'].min()
    maxY = dataFrame['y'].max()
    minY = dataFrame['y'].min()
    dataFrame['x'] = pd.to_numeric(dataFrame['x'], downcast="float")
    dataFrame['y'] = pd.to_numeric(dataFrame['y'], downcast="float")
    for i in range(0,int(dataFrame.size / 2)):
        dataFrame.at[i,'x'] = (dataFrame.at[i,'x'] - minX )/(maxX - minX)
        dataFrame.at[i,'y'] = (dataFrame.at[i,'y'] - minY )/(maxY - minY)
    return dataFrame

#shuffling data for better precision
def shuffeling(dataFrame):
    for i in range(0,int(dataFrame.size/2)):
        r1 = random.randint(0,dataFrame.size/2 - 1)
        r2 = random.randint(0,dataFrame.size/2 - 1)
        x = dataFrame.at[r1,'x']
        y = dataFrame.at[r1,'y']
        dataFrame.at[r1,'x'] = dataFrame.at[r2,'x']
        dataFrame.at[r1,'y'] = dataFrame.at[r2,'y']
        dataFrame.at[r2,'x'] = x
        dataFrame.at[r2,'y'] = y
    return dataFrame

# plotting raw data
def plotData(x,y):
    plt.figure()
    plt.plot(x,y,"og")
    plt.xlabel('X',color="orange")
    plt.ylabel('Y',color="orange")
    plt.title('plot of all signal data' )
    plt.legend(loc='best')
    plt.show()

#splitting data into train and test datasets
def train_test_split(data,ratio):
    X = (data['x'])
    s = int(ratio * len(X))
    x_train = X[:s]
    x_test = X[s:]
    Y = (data['y'])
    s = int(ratio * len(Y))
    y_train = Y[:s]
    y_test = Y[s:]
    return x_train,y_train,x_test,y_test

def Q1_part2():
    data = pd.read_csv('./p1_dataset/data1_Signal.csv')
    data = shuffeling(data)
    data = normalizer(data)
    x_train, y_train, x_test, y_test = train_test_split(data,0.7)
    order = input("Enter order: ")
    iterations = [1000,10000]
    errors = ["MSE","RMSE","MAE"]
    fig, axs = plt.subplots(2, 3, constrained_layout=True,figsize=(20,10))
    fig_cost, axs_cost = plt.subplots(2, 3, constrained_layout=True,figsize=(20,10))
    fig_pace, axs_pace = plt.subplots(2, 3, constrained_layout=True,figsize=(20,10))
    for iteration in iterations:
        for error in errors:
            PR = PolynomialRegression(x_train, y_train, x_test, y_test)
            theta = PR.fit(method='gradient_descent', order=int(order), tol=10 ** -3, numIters=iteration, learningRate=1,
                           error=error)
            PR.plot_predictedPolyLine(axs,iterations.index(iteration), errors.index(error))
            PR.plotCost(axs_cost,iterations.index(iteration), errors.index(error))
            PR.plotPace(axs_pace,iterations.index(iteration), errors.index(error))
    plt.show()

# Q1_part2()

def Q1_part3():
    data = pd.read_csv('./p1_dataset/data1_Signal.csv')
    data = shuffeling(data)
    data = normalizer(data)
    x_train, y_train, x_test, y_test = train_test_split(data, 0.7)
    orders = [3,5,7]
    train_MSEs = []
    test_MSEs = []
    train_RMSEs = []
    test_RMSEs = []
    train_MAEs = []
    test_MAEs = []
    fig, axs = plt.subplots(3, constrained_layout=True, figsize=(8, 10))
    for order in orders:
        PR = PolynomialRegression(x_train, y_train, x_test, y_test)
        theta = PR.fit(method='normal_equation', order=order)
        PR.plot_predictedPolyLine_normalEq(axs,orders.index(order))
        MSE = PR.compute_MSE(x_train, y_train, theta.theta)
        MSE2 = PR.compute_MSE(x_test, y_test, theta.theta)
        RMSE = PR.compute_RMSE(x_train, y_train, theta.theta)
        RMSE2 = PR.compute_RMSE(x_test, y_test, theta.theta)
        MAE = PR.compute_MAE(x_train, y_train, theta.theta)
        MAE2 = PR.compute_MAE(x_test, y_test, theta.theta)
        train_MSEs.append(MSE)
        test_MSEs.append(MSE2)
        train_RMSEs.append(RMSE)
        test_RMSEs.append(RMSE2)
        train_MAEs.append(MAE)
        test_MAEs.append(MAE2)
    plt.show()
    #plotting MSE for train and test vs orders
    plt.plot(orders, train_MSEs, '-o', color="c", label="MSE for Train")
    plt.plot(orders, test_MSEs, '-o', color="m", label="MSE for Test")
    plt.xlabel("order")
    plt.ylabel("MSE")
    plt.legend(loc='best')
    plt.show()

    #plotting RMSE for train and test vs orders
    plt.plot(orders, train_RMSEs, '-o', color="c", label="RMSE for Train")
    plt.plot(orders, test_RMSEs, '-o', color="m", label="RMSE for Test")
    plt.xlabel("order")
    plt.ylabel("RMSE")
    plt.legend(loc='best')
    plt.show()

    #plotting MAE for train and test vs orders
    plt.plot(orders, train_MAEs, '-o', color="c", label="MAE for Train")
    plt.plot(orders, test_MAEs, '-o', color="m", label="MAE for Test")
    plt.xlabel("order")
    plt.ylabel("MAE")
    plt.legend(loc='best')
    plt.show()

# Q1_part3()

def Q1_part4():
    data = pd.read_csv('./p1_dataset/data1_Signal.csv')
    data = shuffeling(data)
    data = normalizer(data)
    Train_MSEs =[]
    Test_MSEs =[]
    x_train, y_train, x_test, y_test = train_test_split(data, 0.7)
    # PR = PolynomialRegression(x_train, y_train, x_test, y_test)
    lambdas = [5,50,500]
    fig, axs = plt.subplots(3, constrained_layout=True, figsize=(8, 10))
    for l in lambdas:
        PR = PolynomialRegression(x_train, y_train, x_test, y_test)
        theta = PR.fit(method='normal_equation_2', order=5 , l =l)
        PR.plot_predictedPolyLine_normalEq(axs, lambdas.index(l))
        MSE = PR.compute_MSE(x_test, y_test, theta.theta)
        MSE2 = PR.compute_MSE(x_train, y_train, theta.theta)
        Test_MSEs.append(MSE)
        Train_MSEs.append(MSE2)
    plt.show()
    plt.plot(lambdas,Train_MSEs,'-o',color="c",label="MSE for Train")
    plt.plot(lambdas,Test_MSEs,'-o',color="m",label="MSE for Test")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend(loc='best')
    plt.show()
# Q1_part4()

