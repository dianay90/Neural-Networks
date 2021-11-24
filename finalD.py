
import numpy as np 
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import csv 


class NeuralNetwork():

    def __init__(self, epochs,l_rate):
        self.epochs = epochs
        self.l_rate = l_rate

        self.parameters = {
            'W1':np.random.randn(128, 784) * np.sqrt(1. / 128),
            'W2':np.random.randn(64, 128) * np.sqrt(1. / 64),
            'W3':np.random.randn(10, 64) * np.sqrt(1. / 10)
        }

    
    def sigmoid2(self, x):
    
        return 1/(1 + np.exp(-x))

    def sigmoid_with_derivative2(self, x):
        sigmoid  = self.sigmoid2(x)
        df = sigmoid * (1 - sigmoid)
        return df
          
    
    def softmax2(self, x):
        #Cite: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python/38250088
        return np.exp(x - x.max())  / np.sum(np.exp(x - x.max()) , axis=0)        

    def softmax_with_derivative2(self,x):
        return np.exp(x - x.max())  / np.sum(np.exp(x - x.max()) , axis=0) * (1 - np.exp(x - x.max())  / np.sum(np.exp(x - x.max()) , axis=0))

  
    def forward_matrix_multiply(self, variableI, variableII):
        finalParam = variableI.dot(variableII)
        return finalParam 
        
    def forward_propagation(self, x_array):
        #for loop or major function call out
        parameter_values = self.parameters

        # multiple weights by prevoius layer activation, then apply act function
        parameter_values['A0'] = x_array

        # matrix multipliy input to first hidden layer, then apply activation function
        parameter_values['Z1'] = self.forward_matrix_multiply(parameter_values['W1'], parameter_values['A0'])
        parameter_values['A1'] = self.sigmoid2(parameter_values['Z1'])

        # matrix multiply input to first hidden layer, then apply activation function

        parameter_values['Z2'] = self.forward_matrix_multiply(parameter_values['W2'], parameter_values['A1'])
        parameter_values['A2'] = self.sigmoid2(parameter_values['Z2'])

        # matrix multiple hidden layer two to output layer, then apply softmax since it's the last year
        parameter_values['Z3'] = self.forward_matrix_multiply(parameter_values['W3'], parameter_values['A2'])
        parameter_values['A3'] = self.softmax2(parameter_values['Z3'])

        return parameter_values['A3']
   
    def make_predictions(self, x_values):
        predictions = list()

        for x in x_values:
            output = self.forward_propagation(x)
            max_output = np.argmax(output)
            predictions.append(max_output)
        
        return predictions
    
    def product_of_vectors(self, error, param): 
        product = np.matrix(error).T.dot(np.matrix(param))
        #np.matrix(error).T.dot(np.matrix(params['A2']))
        return product

    def backward_propogation_update_weights(self, updated_output):
        parameter_values = self.parameters
        updated_weights = dict()

        error = self.softmax_with_derivative2(parameter_values['Z3']) * updated_output * 2
        updated_weights['W3'] = np.matrix(error).T.dot(np.matrix(parameter_values['A2']))

        # Calculate W2 update
        error =  self.sigmoid_with_derivative2(parameter_values['Z2']) *parameter_values['W3'].T.dot(error) 
        updated_weights['W2'] = np.matrix(error).T.dot(np.matrix(parameter_values['A1']))

        # Calculate W1 update
        error =  self.sigmoid_with_derivative2(parameter_values['Z1']) * parameter_values['W2'].T.dot(error)
        updated_weights['W1'] = np.matrix(error).T.dot(np.matrix(parameter_values['A0']))
        
        # Update weights 
        self.parameters['W3'] = self.parameters['W3'] - self.l_rate * updated_weights['W3']
        self.parameters['W3'] =  np.squeeze(np.asarray( self.parameters['W3']))
       
        self.parameters['W2'] = self.parameters['W2'] - self.l_rate * updated_weights['W2']
        self.parameters['W2'] =  np.squeeze(np.asarray( self.parameters['W2']))
        
        self.parameters['W1'] = self.parameters['W1'] - self.l_rate * updated_weights['W1']
        self.parameters['W1'] =  np.squeeze(np.asarray( self.parameters['W1']))

        

    def detect_accuracy(self, xvalue, yvalue):

        predicts_list = list()

        for x, y in zip(xvalue, yvalue):

            output_prediction = np.argmax(self.forward_propagation(x))

            if output_prediction == np.argmax(y):
                predicts_list.append(True)
            else: 
                predicts_list.append(False)
        
        return np.mean(predicts_list)
  

    def cross_entropy_loss(self, predictions, training_labels): 
        return np.mean(training_labels * np.log(predictions.T))

    def train_network(self, x_training_data, y_training_data, x_value, y_value):

        init_time = time.time()

        for epoch_iteration in range(self.epochs):
            for x,y in zip(x_training_data, y_training_data):
                forward_output = self.forward_propagation(x)
                loss= self.cross_entropy_loss(forward_output,y)
                update_output = (forward_output- y)/forward_output.shape[0]
                self.backward_propogation_update_weights(update_output)

            #After each round, see if accuracy has improved
            accuracy = self.detect_accuracy(x_value, y_value)

            update_epoch_iteration = epoch_iteration + 1 
            new_time = time.time() - init_time 
            new_time = str(round(new_time, 2))
            update_accuracy = accuracy * 100 
            update_accuracy = str(round(update_accuracy, 2))
            loss = str(round(loss,2))

            print("Epoch: " + str(update_epoch_iteration) + "\n"
            "Time: " + str(new_time) + "\n"
            "Accuracy: " +str(update_accuracy) +'%'+ "\n"
            "Loss: "  + str(loss) + "\n")

            
    def write_predictions(self, pred):
        currentDirectory = os.getcwd()
        fullPath = os.path.join(currentDirectory,"test_predictions.csv")
        
        if not os.path.exists(fullPath):

            with open(fullPath, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter = '\n')
                csvwriter.writerows(pred)

def get_data():
    Xtest=pd.read_csv("/home/dianayoh/homework3/homework3_ai/test_image.csv",header=None)
    ytest=pd.read_csv("/home/dianayoh/homework3/homework3_ai/test_label.csv",header=None)
    Xtrain=pd.read_csv("/home/dianayoh/homework3/homework3_ai/train_image.csv",header=None)
    ytrain=pd.read_csv("/home/dianayoh/homework3/homework3_ai/train_label.csv",header=None)

    #Reshape to 28x28
    X_train=np.array(Xtrain)[:].reshape((-1, 28, 28))
    X_test=np.array(Xtest)[:].reshape((-1, 28, 28))

    #Divide by 255
    X_test = (X_test/255).astype('float32')
    X_train = (X_train/255).astype('float32')
    
    #Grab labels
    y_train=np.array(ytrain[0])
    y_test=np.array(ytest[0])
  
    rand=np.arange(60000)
    np.random.shuffle(rand)

    #train_no=rand[:50000]
    first_10K = rand [0:10000]
    second_10K = rand[10000:20000]

    X_train, X_val = X_train[first_10K], X_train[second_10K]
    y_train, y_val= y_train[first_10K], y_train[second_10K]
 
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

def one_hot(y_training): 
    temp_y_list = list()
    # return numpy array of one hot 
    for i in y_training: 
        list_item = [0]*10
        list_item[i] = 1
        temp_y_list.append(list_item)
    return np.array(temp_y_list)



if __name__ == "__main__":
 
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()

    y_train = list(y_train)
    y_val = list(y_val)
    y_test = list(y_test)
   
    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_test = one_hot(y_test)

    epoch = 110 #110
    l_rate = .003
    
    dnn = NeuralNetwork(epoch,l_rate)

    dnn.train_network(X_train, y_train, X_val, y_val)

    pred = dnn.make_predictions(X_test)
    dnn.write_predictions(pred)





