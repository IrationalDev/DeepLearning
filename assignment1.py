from ctypes import sizeof
from mimetypes import init
import random
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


#test
def load_data():
    N=500
    gq = sklearn.datasets.make_gaussian_quantiles(mean=None, cov = 0.7, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    return gq

def make_graphs(data, y_true):
    # create a figure and axis
    fig, ax = plt.subplots()

    colors = {0:'r', 1:'b'}
    for i in range(len(y_true)):
        ax.scatter(data[i][0], data[i][1], color=colors[y_true[i]])

    # set a title and labels
    ax.set_title('Random Dataset')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def sigmoid(x):
    sigmoid = np.zeros(x.size)
    for i in range(0, x.size):
        sigmoid[i] = 1/(1+ np.exp(-x[i]))
    return sigmoid


def initialize_weights(N, M):
    w = np.random.uniform(low = -1, high = 1, size = (N,M))
    return w

def forward_pass(w1, w2, x):
    y = np.dot(w1, x)
    y = sigmoid(y)
    y = np.dot(w2, y)
    y = sigmoid(y)
    return y

def backward_pass(w, x):
    return
    


#C:\Users\wopke\OneDrive\Documents\RUG\DeepLearning

#use minibatching

def main():
    #Call functions here
    #random.seed(8)
    gq = load_data()
    data = gq[0]
    y_true = gq[1]
    N_weights = 20
    layer_size = 2
    #initialize first layer
    w1 = initialize_weights(N_weights, layer_size)
    w2 = initialize_weights(1, N_weights)
    print(forward_pass(w1, w2, data[0]))

    y_out = forward_pass(w1, w2, data[1])

    make_graphs(data, y_true)
    print("data:", data)
    
    

if __name__ == "__main__":
    main()
