from ctypes import sizeof
from mimetypes import init
import random
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


<<<<<<< HEAD
#this is a comment
=======
#test
>>>>>>> b3bf124ff436179c63190b50332eb0c8bc3475ac
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

def sigmoidderivative(z):
    deriv = np.zeros(z.size)
    for i in range(0, z.size):
        deriv[i] = 1/(1+ np.exp(-z[i])) * (1-1/(1+ np.exp(-z[i])))
    return deriv


def initialize_weights(N, M):
    w = np.random.uniform(low = -1, high = 1, size = (N,M))
    return w

def forward_pass(w1, w2, x):
    u1 = np.dot(w1, x)
    h1 = sigmoid(u1)
    u2 = np.dot(w2, h1)
    y = sigmoid(u2)
    return y

def backward_pass(w1, w2, x):
    #we want to get dy/dW
    # dy/dw1 = dsigmoid(u2)/du2 * dW2h1/dh1 * dsigmoid(u1)/du1 * dW1'x/dW1
    # dy/dw2 = dy/du2 * du2/dW2 = dsigmoid(u2)/du2 * (d(W2h1)/dh1 = W2?)
    u1 = np.dot(w1, x)
    h1 = sigmoid(u1)
    u2 = np.dot(w2, h1)
    print(w2.size, w2.shape)
    dw1 = sigmoidderivative(u2) * w2.transpose() * sigmoidderivative(u1) * w1.transpose()
    dw2 = sigmoidderivative(u2) * w2.transpose()
    return dw1, dw2
    
def update_weights(w1, w2, dw1, dw2, y, ytrue):
    w1 = w1 - (y-ytrue) * dw1
    w2 = w2 - (y-ytrue) * dw2
    return w1, w2


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
    #print(forward_pass(w1, w2, data[0]))

    y_out = forward_pass(w1, w2, data[1])
    dw1, dw2 = backward_pass(w1, w2, data[1])
    w1, w2 = update_weights(w1, w2, dw1, dw2, y_out, y_true)

    print(w1, w2)

    #make_graphs(data, y_true)
    #print("data:", data)
    
    

if __name__ == "__main__":
    main()
