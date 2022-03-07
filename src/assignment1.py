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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf


class Neural_Network:
    def __init__(self, data, input_size, output_size, hidden_layers):
        self.data = data
        self.layers = input_size + hidden_layers + output_size
        self.amount_layers = len(self.layers)
        ###initialize Biases and Weights
        self.Bias = []
        for i in range (len(self.layers)-1):
            self.Bias.append(np.zeros(self.layers[i+1]))
        self.Weights = []
        for i in range (len(self.layers)-1):
            self.Weights.append((np.random.random((self.layers[i], self.layers[i+1]))*2)-1)


    def forward_pass(self, x):
        U = []
        H = []
        for i in range(len(self.layers)-1):
            if i == 0:
                u = np.dot(np.transpose(self.Weights[i]), x) + self.Bias[i]
            else:
                u = np.dot(np.transpose(self.Weights[i]), h) + self.Bias[i]
            h = sigmoid(u)
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
                layer_error = (H[i] - y) * sigmoidderivative(U[i])
            change_Bias[i] = layer_error
            if i==0:
                change_Weights[i] = np.outer(x, np.transpose(layer_error))
            else:
                change_Weights[i] = np.outer(H[i-1], np.transpose(layer_error))

        return change_Bias, change_Weights 

    def gradient_descent(self, total_change_bias, total_change_weights, old_total_change_bias, old_total_change_weights, batch_size, reg_const, rho):
        
        for i in range(self.amount_layers - 1):
            if old_total_change_weights == 0:
                self.Weights[i] = self.Weights[i] - (reg_const/batch_size) * total_change_weights[i]                
                self.Bias[i] = self.Bias[i] - (reg_const/batch_size) * total_change_bias[i]
            else:
                self.Weights[i] = self.Weights[i] - (reg_const/batch_size) * total_change_weights[i] - rho * ((reg_const/batch_size) *old_total_change_weights[i])
                self.Bias[i] = self.Bias[i] - (reg_const/batch_size) * total_change_bias[i] - rho * ((reg_const/batch_size) *old_total_change_bias[i]) 


    #train_network class manages the training of the network
    def train_network(self, batch_size, iterations, rho):
        #initialization
        loss = []
        accuracy = []
        reg_const = 0.8
        #loop over each epoch
        for j in range(iterations):
            total_change_bias = []
            total_change_weights = []
            total_sse = 0.0
            total_correct_predictions = 0
            data_size = len(self.data[1])
            
            #sample_numbers is only important when mini-batching with batch_size, which we eventually don't do
            sample_numbers = np.random.choice(data_size, batch_size, replace = False)
            #Loop handling backpropagation for each sample
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
            #note that the SSE/batch size = MSE
            loss.append(total_sse/batch_size)
            accuracy.append((total_correct_predictions)/batch_size)

            if(j==0):
                Neural_Network.gradient_descent(self, total_change_bias, total_change_weights, 0, 0, batch_size, reg_const, rho)
            else:
                Neural_Network.gradient_descent(self, total_change_bias, total_change_weights, old_total_change_bias, old_total_change_weights, batch_size, reg_const, rho)
            old_total_change_weights = total_change_weights
            old_total_change_bias = total_change_bias

        return loss, accuracy

    ###classify uses the feedforward algorithm to classify the data, returns accuracy
    def classify(self, x, y):
        U = []
        H = []
        accuracy = 0
        for j in range(len(y)):
            for i in range(len(self.layers)-1):
                if i == 0:
                    u = np.dot(np.transpose(self.Weights[i]), x[j]) + self.Bias[i]
                else:
                    u = np.dot(np.transpose(self.Weights[i]), h) + self.Bias[i]
                h = sigmoid(u)
                U.append(u)
                H.append(h)
            if np.round(H[-1])==y[j]:
                accuracy = accuracy + 1
        accuracy = accuracy / len(y)
        return accuracy


def plot_data(loss, accuracy, kerasloss, kerasaccuracy):
    plt.plot(loss)
    plt.title('Model loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylim([0, 0.5])
    plt.savefig('../ourNN_loss.jpg')
    plt.clf()

    plt.plot(accuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../ourNN_accuracy.jpg')
    plt.clf()

    plt.plot(kerasloss)
    plt.title('Model loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylim([0, 0.5])
    plt.savefig('../keras_loss.jpg')
    plt.clf()

    plt.plot(kerasaccuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../keras_accuracy.jpg')
    plt.clf()

    plt.plot(kerasloss, 'b')
    plt.plot(loss, 'g')
    plt.title('Model loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylim([0, 0.5])
    plt.savefig('../both_loss.jpg')
    plt.clf()
    
    plt.plot(kerasaccuracy, 'b')
    plt.plot(accuracy, 'g')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../both_accuracy.jpg')
    plt.clf()

def load_data():
    N=500
    gq = sklearn.datasets.make_gaussian_quantiles(mean=None, cov = 0.7, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    return gq

def make_graphs(data, y_true):
    fig, ax = plt.subplots()

    colors = {0:'r', 1:'b'}
    for i in range(len(y_true)):
        ax.scatter(data[i][0], data[i][1], color=colors[y_true[i]])

    ax.set_title('Random Dataset')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(f'../Data_visualization.jpg')
    plt.clf()


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


def keras_learning(data, input_size, output_size, hidden_layers, Batch_Size, Epochs, optimizer):
    model = Sequential()
    model.add(Dense(input_size[0], activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1.00, maxval=1.00)))
    for i in range(len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1.00, maxval=1.00)))
    model.add(Dense(output_size[0], activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1.00, maxval=1.00)))

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(data[0], data[1], batch_size = Batch_Size, epochs = Epochs, verbose=0)
    return history, model

def split_data(data, factor):
    x = data[0]
    y = data[1]

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x, y, train_size = factor, shuffle=True)
    data_train = [np.zeros(z.shape) for z in [xtrain, ytrain]]
    data_test = [np.zeros(z.shape) for z in [xtest, ytest]]
    data_train[0] = xtrain
    data_train[1] = ytrain
    data_test[0] = xtest
    data_test[1] = ytest
    return  data_train, data_test


def main():
    ###Data and variables
    data = load_data()
    make_graphs(data[0], data[1])

    ###Adapt the layers to change the shape of the neural network###
    input_size = [2]
    output_size = [1]
    hidden_layers = [10]
    Iteration_Epochs = 3000

    ###10 times model run###
    loss = np.zeros(Iteration_Epochs)
    accuracy = np.zeros(Iteration_Epochs)
    kerasloss = np.zeros(Iteration_Epochs)
    kerasaccuracy = np.zeros(Iteration_Epochs)

    runs = 10
    overall_NN_test_acc = []
    overall_keras_test_acc = []
    overall_NN_time = []
    overall_keras_time = []

    for i in range(runs):
        ### print("currently in run", i) ###uncomment for some feedback on the progress
        
        ###split the data new for every run
        data_train, data_test = split_data(data, 0.8)
        Batch_Size = len(data_train[0])

        ###run the models, collect the computation time and their accuracy and loss for plotting
        start_time = time.time()
        NN = Neural_Network(data_train, input_size, output_size, hidden_layers)
        loss2, accuracy2 = NN.train_network(Batch_Size, Iteration_Epochs, 0.2)
        NNtime = time.time() - start_time

        start_time = time.time()
        history, model = keras_learning(data_train, input_size, output_size, hidden_layers, Batch_Size, Iteration_Epochs, optimizer = SGD(learning_rate=0.8, momentum=0.2))
        kerastime = time.time() - start_time

        ###process collected accuracy and loss for plotting to average over the runs
        kerasloss2 = history.history['loss']
        kerasaccuracy2 = history.history['accuracy']
        loss = [a + b for a, b in zip(loss, loss2)]
        accuracy = [a + b for a, b in zip(accuracy, accuracy2)]
        kerasloss = [a + b for a, b in zip(kerasloss, kerasloss2)]
        kerasaccuracy = [a + b for a, b in zip(kerasaccuracy, kerasaccuracy2)]
        overall_NN_time.append(NNtime)
        overall_keras_time.append(kerastime)

        ###collect validation results on testing data
        testhistory = model.evaluate(data_test[0], data_test[1], verbose = 0)
        overall_keras_test_acc.append(testhistory[1])
        testaccuracy = NN.classify(data_test[0], data_test[1])
        overall_NN_test_acc.append(testaccuracy)

    ###print the validation results
    print("keras", np.std(overall_keras_test_acc), np.mean(overall_keras_test_acc))
    print("NN", np.std(overall_NN_test_acc), np.mean(overall_NN_test_acc))
    print("kerastime", np.std(overall_keras_time), np.mean(overall_keras_time))
    print("NNtime", np.std(overall_NN_time), np.mean(overall_NN_time))
    
    ###process collected accuracy and loss for plotting to average over the runs
    loss = [a / runs for a in loss]
    accuracy = [a / runs for a in accuracy]
    kerasloss = [a / runs for a in kerasloss]
    kerasaccuracy = [a / runs for a in kerasaccuracy]
    plot_data(loss, accuracy, kerasloss, kerasaccuracy)



    ##Gridsearch###
    data_train, data_test = split_data(data, 0.8)
    Batch_Size = len(data_train[0])

    input_size = [2]
    output_size = [1]
    Iteration_Epochs = 3000

    hidden_layerss = [[6],[10],[4,4],[6,6], [3,3,3]]
    SGDs = [SGD(learning_rate=0.8, momentum=0.2), SGD(learning_rate=0.9, momentum=0.2), SGD(learning_rate=1, momentum=0.2), SGD(learning_rate=2, momentum=0.2)]
    for hidden_layers in hidden_layerss:
        for optimizer in SGDs:
            history, model = keras_learning(data_train, input_size, output_size, hidden_layers, Batch_Size, Iteration_Epochs, optimizer = optimizer)
            testhistory = model.evaluate(data_test[0], data_test[1], verbose = 0)
            print(hidden_layers, optimizer.learning_rate, testhistory)

    


if __name__ == "__main__":
    main()
