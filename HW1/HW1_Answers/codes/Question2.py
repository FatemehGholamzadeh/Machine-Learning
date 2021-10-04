###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')

url='./p2_dataset/data2_house_data.csv'
data = pd.read_csv(url)

#plotting raw data according to lat and long
def plotData(data):
    plt.figure(figsize=(20,10))
    y = data["lat"]
    x = data["long"]
    plt.plot(x, y,'o',color='darkcyan')
    plt.xlabel('long')
    plt.ylabel('lat')
    plt.title('Coordinates of Houses' )
    plt.legend(loc='best')
    plt.show()

#plotting heatmap of features
def plot_heatmap(data):
    plt.figure(figsize=(15, 7))
    heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()

#find features to drop
def drop_features():
    cor_matrix = data.corr()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    print(upper_tri)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.76)]
    print(to_drop)

#splitting data to train and test sets
def train_test_split(X,Y,ratio):
    s = int(ratio * len(Y))
    x_test = X[s:]
    x_train = X[:s]
    y_train = Y[:s]
    y_test = Y[s:]
    return x_train,y_train,x_test,y_test

def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error

#gradient_descent finction
def gradient_descent(x_train,x_test, y_train,y_test, theta, alpha, iters):
    cost_array_train = np.zeros(iters)
    cost_array_test = np.zeros(iters)
    thetas = []
    m = y_train.size
    for i in range(iters):
        cost_train, error = cost_function(x_train, y_train, theta)
        cost_test, error2 = cost_function(x_test, y_test, theta)
        theta = theta - (alpha * (1/m) * np.dot(x_train.T, error))
        cost_array_train[i] = cost_train
        cost_array_test[i] = cost_test
        thetas.append(theta)
    return theta, cost_array_train,cost_array_test,thetas

#gradient_descent finction with reqularization and lambda
def gradient_descent_with_regularization(x_train,x_test, y_train,y_test, theta, alpha, iters,lambda_):
    cost_array_train = np.zeros(iters)
    cost_array_test = np.zeros(iters)
    thetas = []
    m = y_train.size
    for i in range(iters):
        cost_train, error = cost_function(x_train, y_train, theta)
        cost_test, error2 = cost_function(x_test, y_test, theta)
        theta = theta - (alpha * ((1/m) * np.dot(x_train.T, error) + lambda_*theta))
        cost_array_train[i] = cost_train
        cost_array_test[i] = cost_test
        thetas.append(theta)
    return theta, cost_array_train,cost_array_test,thetas

def normal_Equation(x, y):
    x = np.array(x)
    ones_ = np.ones(x.shape[0])
    x = np.c_[ones_, x]
    y = np.array(y)
    thetas = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return thetas

def predict(x,thetas):
    try:
        x = np.array(x)
        ones_ = np.ones(x.shape[0])
        x = np.c_[ones_, x]
        result = np.dot(x,thetas)
        return result
    except Exception as e:
        raise e

def plotChart(iterations, cost_train ,cost_test):
    plt.figure()
    plt.plot(np.arange(iterations), cost_train, 'g',label = "cost of train")
    plt.plot(np.arange(iterations), cost_test, 'orangered',label = "cost of test")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs Iterations')
    plt.legend(loc = 'best')
    plt.show()

def plot_pace(iterations, thetas):
    plt.figure()
    paces=[]
    for i in range(len(thetas)):
        paces.append(thetas[i][1])
    plt.plot(np.arange(iterations), paces, 'g')
    plt.xlabel('Iterations')
    plt.ylabel('Pace Size')
    plt.title('Pace Size vs Iterations')
    plt.legend(loc = 'best')
    plt.show()

def part4(data):
    data = (data.loc[:, data.columns != 'date'])
    data = (data - data.mean()) / data.std()

    # Extract data into X and y
    X = (data.loc[:, data.columns != 'price'])
    X = (X.loc[:, X.columns != 'date'])

    # X = data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long,sqft_living15','sqft_lot15']]
    # X = data[['id','bedrooms','bathrooms','sqft_lot','floors','waterfront','view','condition','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]
    y = data['price']
    x_train, y_train, x_test, y_test = train_test_split(X,y,0.7)

    # Add a 1 column to the start to allow vectorized gradient descent
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]

    # Add a 1 column to the start to allow vectorized gradient descent
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    # Set hyperparameters
    alpha = 0.005
    iterations = 1000

    # Initialize Theta Values to 0
    theta = np.zeros(x_train.shape[1])
    initial_cost, _ = cost_function(x_train, y_train, theta)

    print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

    # Run Gradient Descent
    theta, cost_train , cost_test,thetas = gradient_descent(x_train,x_test,y_train,y_test,theta, alpha, iterations)

    # Display cost chart
    plotChart(iterations, cost_train,cost_test)

    #Display cost chart
    plot_pace(iterations,thetas)

    final_cost, _ = cost_function(x_train, y_train, theta)
    test_cost,_= cost_function(x_test, y_test, theta)
    print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))
    print("Test cost : " + str(test_cost))

    # X2 = (data.loc[:, data.columns != 'price'])
    X2 = (data.loc[:, data.columns != 'date'])
    sample = X2.sample(50)
    ys = sample['price']
    sample =(sample.loc[:, sample.columns != 'price'])
    sample2 = np.c_[np.ones(sample.shape[0]), sample]
    sample3 = np.dot(sample2, theta.T)
    plt.plot(sample.index,sample3,'o',color='c',label='predicted price')
    plt.plot(sample.index,ys,'o',color='m',label='real price')
    plt.xlabel('index')
    plt.ylabel('price')
    plt.legend(loc='best')
    plt.show()

# part4(data)


def part5(data):
    data = (data.loc[:, data.columns != 'date'])

    #normalizing data
    data = (data - data.mean()) / data.std()

    # Extract data into X and y
    X = (data.loc[:, data.columns != 'price'])
    X = (X.loc[:, X.columns != 'date'])
    # X = data[['bedrooms','bathrooms','sqft_living','floors','waterfront','view','grade','sqft_above','sqft_basement','yr_renovated','lat','sqft_living15']]
    y = data['price']

    x_train, y_train, x_test, y_test = train_test_split(X, y, 0.7)
    thetas = normal_Equation(x_train,y_train)
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    cost_train,_ = cost_function(x_train,y_train,thetas)
    cost_test , _ = cost_function(x_test,y_test,thetas)
    print(cost_train)
    print(cost_test)

    #sample 50 data points to show predicted price
    X2 = (data.loc[:, data.columns != 'date'])
    sample = X2.sample(50)
    ys = sample['price']
    sample = (sample.loc[:, sample.columns != 'price'])
    sample2 = np.c_[np.ones(sample.shape[0]), sample]
    sample3 = np.dot(sample2, thetas.T)
    plt.plot(sample.index, sample3, 'o', color='c', label='predicted price')
    plt.plot(sample.index, ys, 'o', color='m', label='real price')
    plt.xlabel('index')
    plt.ylabel('price')
    plt.legend(loc='best')
    plt.show()

# part5(data)