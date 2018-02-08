from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import normalize


def createSubgroup(classes, images, labels):
    # code for creating subclasses from training set
    m = len(images)  # length of subset
    n = len(images[0])  # 784
    subgroup = []
    for j in range(0, classes):
        subgroup.append([])

    for i in range(m):
        cur = labels[i]
        if cur == 0:
            subgroup[0].append(images[i])

        elif cur == 1:
            subgroup[1].append(images[i])

        elif cur == 2:
            subgroup[2].append(images[i])

        elif cur == 3:
            subgroup[3].append(images[i])

        elif cur == 4:
            subgroup[4].append(images[i])

        elif cur == 5:
            subgroup[5].append(images[i])

        elif cur == 6:
            subgroup[6].append(images[i])

        elif cur == 7:
            subgroup[7].append(images[i])

        elif cur == 8:
            subgroup[8].append(images[i])

        elif cur == 9:
            subgroup[9].append(images[i])
    return subgroup

def eigenfaces(subgroup, d):
    # this is the initial component analysis
    m = len(subgroup)
    n = len(subgroup[0])

    mu = sum(subgroup)
    utot = 1/m * mu

    x = []
    for i in range(0, m):
        x.append(subgroup[i] - utot)
    xarray = np.asarray(x)
    xarray = np.matrix.transpose(xarray)
    [u, s, v] = np.linalg.svd(xarray)
    ureduce = u[:, 0:d]
    ureduce = normalize(ureduce)
    # print(type(x))
    # return ureduce, utot
    xlist = []
    for i in range(0, m):  # length of subset
        standDev = np.std(subgroup[i])
        mean = 1/standDev*subgroup[i]-utot
        xlist.append(mean)
    xmat = np.vstack(xlist)
    utrap = np.matrix.transpose(ureduce)
    xmat = np.matrix.transpose(xmat)
    wmat = np.mat(utrap) * np.mat(xmat)
    return wmat





# def applyUreduce(ureduce,subset,utot):
#     m = len(subset)  # length of subset
#     n = len(subset[0])  # 784
#     xlist = []
#     for i in range(0, m):  # length of subset
#         standDev = np.std(subset[i])
#         mean = 1/standDev*subset[i]-utot
#         xlist.append(mean)
#     xmat = np.vstack(xlist)
#     utrap = np.matrix.transpose(ureduce)
#     xmat = np.matrix.transpose(xmat)
#     wmat = np.mat(utrap) * np.mat(xmat)
#     # print(wmat.shape)
#     return wmat


def createBagOfWords(classes,subgroup,k):
    bag = []
    for i in range(0,classes):
        curReduce = eigenfaces(subgroup[i], k)
        bag.append(curReduce)
        # gets ureduce for each number
    return(bag)

def nearestNeighbors(testSet, trainSet, numTrain, numTest):
    d = []
    trainGround = []
    testGround = []
    results = []
    for j in range(0,len(numTrain)):
        trainGround.append(len(numTrain[j]))

    print(trainGround)
    # for k in range(0,len(numTest)):
    #     testGround.append(len(numTest[k]))


    # num = [0] * 10
    # for i in range (0,len(testSet)):
    #     for j in range(i,len(trainSet)):
    #         d[j] = normalize(testSet[:, i]-trainSet[:, i])
    #     nearest = sorted(d)
    #     nearestIdx = np.argmin(d)
    #     d1 = nearest[0]
    #     d2 = nearest[1]
    #
    #     ratio = d1/d2
    #     if ratio < 1:
    #         if numTrain[i] == numTest[nearestIdx]:
    #             num[numTrain[i]] = num[numTrain[i]]+1
    #
    # for k in range(0,10):
    #     results[k] = (num(k)/testGround(k));
    #
    # conf = 1/10*sum(results);


if __name__ == '__main__':
    k = 9
    classes = 10
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    trainIm = mnist.train.images
    trainlabels = mnist.train.labels

    testIm = mnist.test.images
    testLabels = mnist.test.labels

    subgroupList = createSubgroup(classes, trainIm, trainlabels)
    # testgroupList = createSubgroup(classes, testIm, testLabels)

    # can do stochastic modeling to get a training model...
    # go for pca analysis of groups, fisher/bag of words approach

    bag = createBagOfWords(classes, subgroupList,k)
    # creates bag of mean faces

    print(len(bag), bag[0].shape)
    # ureduce = eigenfaces(subgroupList[0], k)
    #
    # trainset = applyUreduce(ureduce, trainIm, utot)
    # testSet = applyUreduce(ureduce, testIm, utot)

    # show image test
    # A = subgroup[4][0]
    # A.resize([28, 28])
    # plt.imshow(A)
    # plt.show()

    #


