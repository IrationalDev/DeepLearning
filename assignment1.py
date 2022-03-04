from ctypes import sizeof
from mimetypes import init
from socket import getfqdn
from ssl import HAS_NEVER_CHECK_COMMON_NAME
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


class Neural_Network:
    def __init__(self, data, input_size, output_size, hidden_layers):
        self.data = data
        self.layers = input_size + hidden_layers + output_size
        self.amount_layers = len(self.layers)
        #initialize Biases
        self.Bias = []
        for i in range (len(self.layers)-1):
            self.Bias.append((np.random.random(self.layers[i+1])*2)-1)
        # self.Bias = np.array(self.Bias)
        #initialize Weights
        self.Weights = []
        for i in range (len(self.layers)-1):
            self.Weights.append((np.random.random((self.layers[i], self.layers[i+1]))*2)-1)
        # print('weight array',self.Weights)
        # print(self.Weights[1])
        # print('bias array', self.Bias)


    def forward_pass(self, x):
        U = []
        H = []
        for i in range(len(self.layers)-1):
            if i == 0:
                u = np.dot(np.transpose(self.Weights[i]), x) + self.Bias[i]
            else:
                u = np.dot(np.transpose(self.Weights[i]), h) + self.Bias[i]
            #print("u", u)
            h = sigmoid(u)
            #print("h", h)
            U.append(u)
            H.append(h)
        return U, H

    def backward_pass(self, x, y, U, H):
        change_Bias = [np.zeros(b.shape) for b in self.Bias]
        change_Weights = [np.zeros(w.shape) for w in self.Weights]
        for i in range (self.amount_layers -2, -1, -1):
            if i != self.amount_layers -2:
                layer_error = sigmoidderivative(U[i]) * (np.dot(self.Weights[i+1], layer_error))
            else:
                #print(U[i] , y , H[i])
                layer_error = (H[i] - y) * sigmoidderivative(U[i])
            change_Bias[i] = layer_error
            if i==0:
                change_Weights[i] == np.outer(x, np.transpose(layer_error))
            else:
                change_Weights[i] = np.outer(H[i-1], np.transpose(layer_error))
        return change_Bias, change_Weights 

    def gradient_descent(self, total_change_bias, total_change_weights, batch_size, reg_const):
        for i in range(self.amount_layers - 1):
            #print(" Weights before:", self.Weights[i])
            self.Weights[i] = self.Weights[i] - (reg_const/batch_size) * total_change_weights[i]
            self.Bias[i] = self.Bias[i] - (reg_const/batch_size) * total_change_bias[i]
            #print(" Weights after:", self.Weights[i])



        

    def train_network(self, batch_size, iterations):
        for j in range(iterations):
            total_change_bias = []
            total_change_weights = []
            total_sse = 0.0
            total_correct_predictions = 0
            reg_const = 0.95
            data_size = len(self.data[1])
            #print("data size", data_size)
            sample_numbers = np.random.choice(data_size, batch_size, replace = False)
            for i in range(batch_size):
                x = self.data[0][sample_numbers[i]]
                y = self.data[1][sample_numbers[i]]
                U, H = Neural_Network.forward_pass(self, x)
                change_bias, change_weights = Neural_Network.backward_pass(self, x, y, U, H)
                if i == 0:
                    total_change_bias = change_bias
                    total_change_weights = change_weights
                else:
                    total_change_bias = np.add(total_change_bias, change_bias)
                    total_change_weights = np.add(total_change_weights, change_weights)
                total_sse = total_sse + np.square(H[-1] - y)
                total_correct_predictions = total_correct_predictions + (np.round(H[-1])==y)
                #print(total_change_weights, change_weights)
            #print("ypred", H[-1])
            #print(total_correct_predictions)
            #print("totalchangew, and bias", total_change_weights[2], total_change_bias[2])
            if j%100 == 0: 
                print("Mean squared error = {} \n Accuracy = {}".format(total_sse/batch_size, (total_correct_predictions*100)/batch_size))
            Neural_Network.gradient_descent(self, total_change_bias, total_change_weights, batch_size, reg_const)
        return





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
    plt.savefig(f'../DeepLearning/{Data_visualization}.jpg')
    plt.show()


def sigmoid(vector):
    sig = np.zeros(len(vector))
    for i in range(len(vector)):
        sig[i] = 1/(1+np.exp(-vector[i]))
    return sig

def sigmoidderivative(vector):
    sigderiv = np.zeros(len(vector))
    for i in range(len(vector)):
        sigderiv[i] = 1/(1+np.exp(-vector[i])) *(1-(1/(1+np.exp(-vector[i]))))
    return sigderiv


def keras_learning(data, input_size, output_size, hidden_layers, batch_size, Epochs):
    optimizer = SGD(learning_rate=0.8)
    model = Sequential()
    model.add(Dense(input_size[0], activation='sigmoid'))
    for i in range(len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation='sigmoid'))
    model.add(Dense(output_size[0], activation='sigmoid'))

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(data[0], data[1], batch_size = batch_size, epochs = Epochs, verbose=0)
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.savefig(f'../DeepLearning/{keras_loss}.jpg')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(f'../DeepLearning/{keras_accuracy}.jpg')
    plt.show()
    return history

def main():
    #Call functions here
    #random.seed(8)
    gq = load_data()
    data = gq[0]
    y_true = gq[1]
    input_size = [2]
    output_size = [1]
    hidden_layers = [6]
    NN = Neural_Network(gq, input_size, output_size, hidden_layers)
    Batch_Size = 500
    Iteration_Epochs = 10
    NN.train_network(Batch_Size, Iteration_Epochs)
    keras_learning(data, input_size, output_size, hidden_layers, Batch_Size, Iteration_Epochs)

    
    
    # print('layers:', NN.layers)

    #initialize first layer



    #make_graphs(data, y_true)
    #print("data:", data)
    
    

if __name__ == "__main__":
    main()
