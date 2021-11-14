    
import numpy as np
import math
import pandas as pd 
from matplotlib import pyplot as plt 


class NeuralNetwork:
    def __init__(self):
        print('te')

    def read_images(self,filename):
        with open(filename) as file: 
            #Separate into sets of 184 
            data= file.read()
        
        dataList = list() 
        dataList = data
        return dataList
    def splitInto784(self, data):
        
        n = 784
        line = data
        finalList =  [line[i:i+n] for i in range(0, len(line), n)]
        return finalList
        
        '''
        line = '1234567890'
        n = 2
        list2 = [line[i:i+n] for i in range(0, len(line), n)]
        print("")
        '''

    def read_labels(self,filename): 
        labelsList = list()
        with open(filename, 'r') as file: 
            lines = file.readlines()
            for item in lines: 
                item = item.rstrip('\n')
                labelsList.append(item)
        return labelsList
    #ReFACTOR
    def init(self,x,y):
        layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
        return layer.astype(np.float32)

    '''
    #forward and backward pass
    def forward_backward_pass(x,y):
        targets = np.zeros((len(y),10), np.float32)
        targets[range(targets.shape[0]),y] = 1
    
        
        x_l1=x.dot(l1)
        x_sigmoid=sigmoid(x_l1)
        x_l2=x_sigmoid.dot(l2)
        out=softmax(x_l2)
    
    
        error=2*(out-targets)/out.shape[0]*d_softmax(x_l2)
        update_l2=x_sigmoid.T@error
        
        
        error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
        update_l1=x.T@error

        return out,update_l1,update_l2 

    '''
    #CITE
    '''
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    '''
    #CITE
    #Sigmoid funstion
    def sigmoid(self,x):
        return 1/(np.exp(-x)+1)    

    #derivative of sigmoid
    def d_sigmoid(self,x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def softmax(self,x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)

    #derivative of softmax
    def d_softmax(self,x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
    #forward and backward pass
    def forward_backward_pass(self,x,y):
        targets = np.zeros((len(y),10), np.float32)
        targets[range(targets.shape[0]),y] = 1
    
        
        x_l1=x.dot(l1)
        x_sigmoid=self.sigmoid(x_l1)
        x_l2=x_sigmoid.dot(l2)
        out=self.softmax(x_l2)
    
    
        error=2*(out-targets)/out.shape[0]*self.d_softmax(x_l2)
        update_l2=x_sigmoid.T@error
        
        
        error=((l2).dot(error.T)).T*self.d_sigmoid(x_l1)
        update_l1=x.T@error

        return out,update_l1,update_l2 


if __name__ == "__main__":
        network = NeuralNetwork()
        testImages = network.read_images("/home/dianayoh/homework3/homework3_ai/test_image.csv")

        List784Test  = network.splitInto784(testImages)
        testLabels = network.read_labels("/home/dianayoh/homework3/homework3_ai/test_label.csv")

        trainImages= network.read_images("/home/dianayoh/homework3/homework3_ai/train_image.csv")

        List784Train = network.splitInto784(trainImages)
        trainLabels = network.read_labels("/home/dianayoh/homework3/homework3_ai/train_label.csv")
        '''
        np.random.seed(42)
        l1=network.init(28*28,128)
        l2=network.init(128,10)

        epochs=10000
        lr=0.001
        batch=128

        losses,accuracies,val_accuracies=[],[],[]

        for i in range(epochs):
            sample=np.random.randint(0,List784Train.shape[0],size=(batch))
            x=List784Test[sample].reshape((-1,28*28))
            y=List784Test[sample]
        

            out,update_l1,update_l2=network.forward_backward_pass(x,y)
        
            category=np.argmax(out,axis=1)
            accuracy=(category==y).mean()
            accuracies.append(accuracy)
            
            loss=((category-y)**2).mean() #test
            losses.append(loss.item())
            
            l1=l1-lr*update_l1
            l2=l2-lr*update_l2
            
            if(i%20==0):    
                X_val=X_val.reshape((-1,28*28))
                val_out=np.argmax(network.softmax(network.sigmoid(X_val.dot(l1)).dot(l2)),axis=1)
                val_acc=(val_out==testLabels).mean()
                val_accuracies.append(val_acc.item())
            if(i%500==0): print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')
        '''