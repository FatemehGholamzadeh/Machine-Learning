###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import style
from Question2 import cost_function,gradient_descent,plotChart,gradient_descent_with_regularization,plot_pace
style.use('ggplot')

def part1():
    url = './p3_dataset/data3_house_data.csv'
    data = pd.read_csv(url)
    data = (data - data.mean()) / data.std()
    # Extract data into X and y
    X = (data.loc[:, data.columns != 'SalePrice'])
    X = (X.loc[:, X.columns != 'id'])
    y = data['SalePrice']
    y_test = y[1000:1200]
    x_test = X[1000:1200]
    # Add a 1 column to the start to allow vectorized gradient descent
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    test_costs = []
    train_costs = []
    for i in range(10, 1001, 10):
        x_train = X[:i]
        y_train = y[:i]

        # Add a 1 column to the start to allow vectorized gradient descent
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]

        # Set hyperparameters
        alpha = 0.005
        iterations = 1000

        # Initialize Theta Values to 0
        theta = np.zeros(x_train.shape[1])
        initial_cost, _ = cost_function(x_train, y_train, theta)

        # Run Gradient Descent
        theta, cost_train, cost_test, thetas = gradient_descent(x_train, x_test, y_train, y_test, theta, alpha,
                                                                iterations)
        final_cost, _ = cost_function(x_train, y_train, theta)
        test_cost, _ = cost_function(x_test, y_test, theta)
        test_costs.append(test_cost)
        train_costs.append(final_cost)
        print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))
        print("Test cost : " + str(test_cost))
        print("***********************************")
    plotChart(100, train_costs, test_costs)

# part1()

def part2():
    url = './p3_dataset/data3_house_data.csv'
    data = pd.read_csv(url)
    data = (data - data.mean()) / data.std()
    X = (data.loc[:, data.columns != 'id'])
    x_test = X[1000:]
    x_test = (x_test.loc[:, x_test.columns != 'SalePrice'])

    # Add a 1 column to the start to allow vectorized gradient descent
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    y_test = data['SalePrice'][1000:]
    sample = X[:1000].sample(50)
    y_train = sample['SalePrice']
    x_train = (sample.loc[:, sample.columns != 'SalePrice'])

    # Add a 1 column to the start to allow vectorized gradient descent
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]

    # Set hyperparameters
    alpha = 0.001
    iterations = 1000

    # Initialize Theta Values to 0
    theta = np.zeros(x_train.shape[1])
    initial_cost, _ = cost_function(x_train, y_train, theta)

    # Run Gradient Descent with_regularization to find optimum lambda
    costs= []
    l = 0.0
    while l <= 5.0:
        theta, cost_train, cost_test, thetas = gradient_descent_with_regularization(x_train, x_test, y_train, y_test,theta, alpha,iterations, l)
        test_cost, _ = cost_function(x_test, y_test, theta)
        costs.append(test_cost)
        print(l)
        print(test_cost)
        print("************")
        l+=0.2
    plt.figure()
    plt.plot(0.2*np.arange(0,25) ,costs, 'o-', color="darkcyan")
    plt.xlabel("lambda")
    plt.ylabel("test cost")
    plt.show()
    theta, cost_train, cost_test, thetas = gradient_descent_with_regularization(x_train, x_test, y_train, y_test,theta, alpha,iterations, 0.8)
    plot_pace(1000,thetas)
    plotChart(1000,cost_train,cost_test)


def part3():
    url = './p3_dataset/data3_house_data.csv'
    data = pd.read_csv(url)
    data = (data - data.mean()) / data.std()

    # X = (data.loc[:, data.columns != 'SalePrice'])
    X = (data.loc[:, data.columns != 'id'])
    x_train = X[:1000]
    x_test = X[1000:]
    x_test = (x_test.loc[:, x_test.columns != 'SalePrice'])

    # Add a 1 column to the start to allow vectorized gradient descent
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    y_test = data['SalePrice'][1000:]
    df_subset = pd.DataFrame()
    means=[]
    variances = []
    for i in range(0,10):
        sample = x_train.sample(100)
        df_subset = df_subset.append(sample)
        y_train = df_subset['SalePrice']
        x_train_2  = (df_subset.loc[:, df_subset.columns != 'SalePrice'])

        # Add a 1 column to the start to allow vectorized gradient descent
        x_train_2 = np.c_[np.ones(x_train_2.shape[0]), x_train_2]
        x_train = x_train.drop(sample.index)

        # Set hyperparameters
        alpha = 0.005
        iterations = 1000

        # Initialize Theta Values to 0
        theta = np.zeros(x_train_2.shape[1])
        initial_cost, _ = cost_function(x_train_2, y_train, theta)

        # Run Gradient Descent
        theta, cost_train, cost_test, thetas = gradient_descent(x_train_2, x_test, y_train, y_test, theta, alpha,
                                                                iterations)
        means.append(np.mean(cost_train))
        print(np.var(cost_test))
        variances.append(np.var(cost_train))
        print("**************************")

    plt.figure()
    plt.plot(100 * np.arange(1, len(means)+1), means,'o-', color="darkcyan")
    plt.xlabel("number of train data")
    plt.ylabel("mean of errors")
    plt.figure()
    plt.plot(100*np.arange(1, len(variances)+1), variances,'o-', color="orangered")
    plt.xlabel("number of train data")
    plt.ylabel("variance of errors")
    plt.show()

# part3()


# part1()


