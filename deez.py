#!/usr/bin/env python
# coding: utf-8

import numpy as np ## For numerical python
import pandas as pd
import time
import os
from tqdm import trange
import matplotlib.pyplot as plt
from IPython.display import clear_output




class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=110, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def predict(self, x_val):

        predictions = []

        for x in x_val:
            output = self.forward_pass(x)
            predictions.append(np.argmax(output))
        
        return predictions

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):

        train_log,val_log = [], []
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            train_log.append(self.compute_accuracy(x_train, y_train))
            val_log.append(self.compute_accuracy(x_val, y_val))
    
            clear_output()
            print("Train accuracy:",train_log[-1])
            print("Val accuracy:",val_log[-1])
            plt.plot(train_log,label='train accuracy')
            plt.plot(val_log,label='val accuracy')
            plt.legend(loc='best')
            plt.grid()
            plt.show()


            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))



def load_dataset():
    import pandas as pd
    Xtest=pd.read_csv("/home/dianayoh/homework3/homework3_ai/test_image.csv",header=None)
    ytest=pd.read_csv("/home/dianayoh/homework3/homework3_ai/test_label.csv",header=None)
    Xtrain=pd.read_csv("/home/dianayoh/homework3/homework3_ai/train_image.csv",header=None)
    ytrain=pd.read_csv("/home/dianayoh/homework3/homework3_ai/train_label.csv",header=None)




    X_train=np.array(Xtrain)[:].reshape((-1, 28, 28))

    X_test=np.array(Xtest)[:].reshape((-1, 28, 28))

    X_test = (X_test/255).astype('float32')
    X_train = (X_train/255).astype('float32')
    y_train=np.array(ytrain[0])
    y_test=np.array(ytest[0])
  
    arrtr= np.random.randint(0,60000,10000)   #randomly select the 10000 images from training.csv
    arrte= np.random.randint(0,60000,10000)   #randomly select the 10000 images from training.csv
     
    X_train, X_val = X_train[arrtr], X_train[arrte]
    y_train, y_val = y_train[arrtr], y_train[arrte]

    X_val = X_train
    y_val = y_train
 
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test
    


def load_datasetv2():
    import pandas as pd
    Xtest=pd.read_csv("/home/dianayoh/homework3/homework3_ai/test_image.csv",header=None)
    ytest=pd.read_csv("/home/dianayoh/homework3/homework3_ai/test_label.csv",header=None)
    Xtrain=pd.read_csv("/home/dianayoh/homework3/homework3_ai/train_image.csv",header=None)
    ytrain=pd.read_csv("/home/dianayoh/homework3/homework3_ai/train_label.csv",header=None)

    #Shuffle stuff

    X_train=np.array(Xtrain)
    np.random.shuffle(X_train)

    X_test=np.array(Xtest)
    np.random.shuffle(X_test)

    y_train=np.array(ytrain[0])
    np.random.shuffle(y_train)

    y_test=np.array(ytest[0])
    np.random.shuffle(y_test)

    #Divide stuff by 255 and transpose
    X_test = (X_test/255).astype('float32').T
    a,b =    X_test.shape
    #X test is a,b where a = 784 rows and 10K columns 
    X_train = (X_train/255).astype('float32').T
    m, n = X_train.shape
    # 784 rows, 60K columns 

    
    '''

    arrtr= np.random.randint(0,60000,10000)   #randomly select the 10000 images from training.csv
    arrte= np.random.randint(0,60000,10000)   #randomly select the 10000 images from training.csv
     
    X_train, X_val = X_train[arrtr], X_train[arrte]
    y_train, y_val = y_train[arrtr], y_train[arrte]

    X_val = X_train
    y_val = y_train
 
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])
    '''
    
    return X_train, y_train,  X_test, y_test
  
#X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
X_train, Y_train,  X_dev, Y_dev = load_datasetv2()
m= 784

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
    
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



'''
y_train = list(y_train)
y_val = list(y_val)
y_test = list(y_test)


y_train_temp, y_val_temp, y_test_temp =  [], [], []

for i in y_train:
  li = [0]*10
  li[i] = 1
  y_train_temp.append(li)


for i in y_val:
  li = [0]*10
  li[i] = 1
  y_val_temp.append(li)

for i in y_test:
  li = [0]*10
  li[i] = 1
  y_test_temp.append(li)


y_train = np.array(y_train_temp)
y_val = np.array(y_val_temp)
y_test = np.array(y_test_temp)


dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(X_train, y_train, X_val, y_val)


pred = dnn.predict(X_test)
dataframe=pd.DataFrame(pd.Series(pred))
dataframe.to_csv(os.getcwd() + "test_predictions.csv", index=False)




'''