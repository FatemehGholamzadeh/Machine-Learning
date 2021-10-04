###############################################
# Student_ID : 99131003                       #
# Name : Fatemeh                              #
# Last Name : Gholamzadeh                     #
# E-mail : fatemeh.gholamzadeh77@gmail.com    #
###############################################
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from random import randint

#training part of one vs all Logostic Regression
def train_phase(images,train_labels):
    labels = []
    classifiers = []
    for i in range(10):
        labels.append([])
    for j in range(10):
        for i in range(len(train_labels)):
            if train_labels[i] != j:
                labels[j].append(0)
            else:
                labels[j].append(1)
        clf = LogisticRegression(random_state=0).fit(images, labels[j])
        classifiers.append(clf)
    return classifiers


#Testing part of one vs all Logostic Regression
def test_phase(classifiers,test_images):
    predict = []
    for image in test_images:
        array = []
        for i in range(len(classifiers)):
            array.append(classifiers[i].predict_proba([image])[0][1])
        max_probability = np.argmax(array)
        predict.append(max_probability)
    return np.array(predict)


#finding Acuuracy
def accuracy_finder(predictions,testLabels):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == testLabels[i]:
            count += 1
    return count / len(testLabels) * 100


#plot digit images
def show_images(classifiers,test_images,test_labels):
    indexes = []
    for i in range(25):
        value = randint(0, len(test_images))
        indexes.append(value)
    num_row = 5
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(2 * num_col, 2 * num_row))
    for i in range(len(indexes)):
        image = test_images[indexes[i]]
        pred = test_phase(classifiers,[image])
        print(pred)
        image = image.reshape((8, 8)).astype("uint8")
        ax = axes[i // num_col, i % num_col]
        ax.imshow(image, cmap='gray')
        # ax.set_title('pre: {}'.format(pred))
        ax.set_title("act=%d, pre=%d" % (test_labels[indexes[i]], pred[0]))
    plt.tight_layout()
    plt.show()


def part1_and_2():

    mnist = load_digits()

    #splitting dataset into train and test parts
    train_images, test_images, train_labels, test_labels = train_test_split(np.array(mnist.data), mnist.target,
                                                                              test_size=0.3, random_state=42)
    # training phase
    classifiers = train_phase(train_images, train_labels)

    # # find error for train data
    predictions = test_phase(classifiers, train_images)
    accuracy = accuracy_finder(predictions, train_labels)
    print("Train Error:", 100 - accuracy,"%")

    # finding error for test data
    predictions = test_phase(classifiers, test_images)
    accuracy = accuracy_finder(predictions, test_labels)
    print("Test Error:",100-accuracy,"%")

    # finding confusion_matrix
    print(confusion_matrix(test_labels, predictions))

    #plot figures
    show_images(classifiers,test_images,test_labels)


if __name__=='__main__':
    part1_and_2()


