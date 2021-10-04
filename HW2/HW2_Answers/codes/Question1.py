###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from random import randint


def euclidean_distance(img_a, img_b):
    #Finds the distance between 2 images: img_a, img_b
    return sum((img_a - img_b) ** 2)

def manhatan_distance(img_a, img_b):
    #Finds the distance between 2 images: img_a, img_b
    return sum(abs(img_a - img_b))

def cosine_similarity(list_1, list_2):
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return 1-cos_sim

def find_majority(labels):
    #Finds the majority class/label out of the given labels
    #defaultdict(type) is to automatically add new keys without throwing error.
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1

    # Finding the majority class.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key

def predict(k, train_images, train_labels, test_images,distance_measure='euclidean'):
    '''
    Predicts the new data-point's category/label by
    looking at all other training labels
    '''
    # distances contains tuples of (distance, label)
    if distance_measure=='euclidean':
        distances = [(euclidean_distance(test_images, image), label)
                    for (image, label) in zip(train_images, train_labels)]
    if distance_measure=='manhatan':
        distances = [(manhatan_distance(test_images, image), label)
                    for (image, label) in zip(train_images, train_labels)]
    if distance_measure=='cosine':
        distances = [(cosine_similarity(test_images, image), label)
                    for (image, label) in zip(train_images, train_labels)]
    # sort the distances list by distances
    #by_distances = sorted(distances, key=lambda distance: distance)
    by_distances = sorted(distances)
    # extract only k closest labels
    k_labels = [label for (_, label) in by_distances[:k]]
    # return the majority voted label
    return find_majority(k_labels)

def predict_all(trainData,trainLabels,valData,valLabels,k,distance):
    i = 0
    total_correct = 0
    acces = []
    predictions = []
    for test_image in valData:
        pred = predict(k, trainData, trainLabels, test_image,distance)
        predictions.append(pred)
        if pred == valLabels[i]:
            total_correct += 1
        acc = (total_correct / (i + 1)) * 100
        acces.append(acc)
        i += 1
    return acces,predictions

def accuracy_finder(predictions,testLabels):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == testLabels[i]:
            count += 1
    return count / len(testLabels) * 100


def find_best_parameters(trainData,trainLabels,valData,valLabels):

    # initialize the values of k for our k-Nearest Neighbor classifier
    kVals = range(1, 30)

    # list of accuracies for each value of k
    accuracies = [[], [], []]

    # list of distances for checking which is better
    distances = ["euclidean", "manhatan", "cosine"]

    # loop over distances and kVals
    for distance in distances:
        for k in range(1, 30):
            # train the classifier with the current value of `k`
            acces,_=predict_all(trainData,trainLabels,valData,valLabels,k,distance)
            score = np.mean(acces)
            # evaluate the model and print the accuracies list
            print("k=%d, accuracy=%.2f%%" % (k, score))
            accuracies[distances.index(distance)].append(score)

    # plot accuracy on validation data for different ks and distances
    for i in range(3):
        plt.plot(kVals, accuracies[i], '-o', label=distances[i])
    plt.xlabel("K")
    plt.ylabel("accuracy on validation data")
    plt.legend(loc='best')
    plt.show()


def show_images(trainData,trainLabels,test_images,test_labels):
    indexes = []
    for i in range(100):
        value = randint(0, len(test_images))
        indexes.append(value)
    num_row = 10
    num_col = 10
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(len(indexes)):
        image = test_images[indexes[i]]
        pred = predict(5, trainData, trainLabels, image, 'euclidean')
        image = image.reshape((8, 8)).astype("uint8")
        ax = axes[i // num_col, i % num_col]
        ax.imshow(image, cmap='gray')
        # ax.set_title('pre: {}'.format(pred))
        ax.set_title("act=%d, pre=%d" % (test_labels[indexes[i]], pred))
    plt.tight_layout()
    plt.show()


def part1():
    mnist = load_digits()
    (train_images, test_images, train_labels, test_labels) = train_test_split(np.array(mnist.data), mnist.target,
                                                                              test_size=0.20, random_state=42)
    # take 25% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(train_images, train_labels, test_size=0.25,
                                                                    random_state=84)
    # Checking sizes of each data split
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(test_labels)))

    # finding best k and distance function -> answer:  k=5, distance=euclidean
    # find_best_parameters(trainData,trainLabels,valData,valLabels)

    accuracies,predictions = predict_all(trainData,trainLabels,test_images,test_labels,5,'euclidean')
    print("Test Accuracy:", accuracy_finder(predictions,test_labels))

    # finding confusion_matrix
    print(confusion_matrix(test_labels,predictions))

    _,predictions= predict_all(trainData,trainLabels,trainData,trainLabels,5,'euclidean')
    print("Train Accuracy:", accuracy_finder(predictions, trainLabels))

    # show_images(trainData, trainLabels, test_images, test_labels)


def part2():
    # load the MNIST digits dataset
    mnist = datasets.load_digits()

    # Training and testing split,
    # 80% for training and 20% for testing
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target,
                                                                      test_size=0.2, random_state=42)

    # take 25% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.25,
                                                                    random_state=84)

    # Checking sizes of each data split
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k
    kVals = range(1, 30)
    accuracies = []

    # loop over kVals
    for k in range(1, 30):
        # train the classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)

        # evaluate the model and print the accuracies list
        score = model.score(valData, valLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # largest accuracy
    # np.argmax returns the indices of the maximum values along an axis
    i = np.argmax(accuracies)
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))

    # Now that I know the best value of k, re-train the classifier
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)

    # Predict labels for the test set
    test_predictions = model.predict(testData)
    print("Test Accuracy:",accuracy_finder(test_predictions,testLabels))

    # Predict labels for the train set
    train_predictions = model.predict(trainData)
    print("Train Accuracy:", accuracy_finder(train_predictions, trainLabels))

    # finding confusion_matrix
    print(confusion_matrix(testLabels, test_predictions))

    # show_images(trainData,trainLabels,testData,testLabels)

if __name__=='__main__':
    part1()
    # part2()

