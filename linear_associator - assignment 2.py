# Kanchi, Gowtham Kumar
# 1002-044-003
# 2022-10-09
# Assignment-02-01
import numpy as np

class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function.lower()

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions)
    def set_weights(self, W):
        
        a,b = W.shape
        if a == self.number_of_nodes and b == self.input_dimensions :
            self.weights = W
        else:
            return -1

    def get_weights(self):
        
        return self.weights

    def predict(self, X):
       
        fnet = np.dot(self.weights,X)
        if self.transfer_function == 'linear':
            out_fn = fnet
        elif self.transfer_function == 'hard_limit':
            out_fn = fnet>=0
            out_fn = np.multiply(out_fn,1)
        return out_fn

    def fit_pseudo_inverse(self, X, y):
       
        self.weights = np.dot(y,np.dot(np.linalg.pinv(np.dot(X.transpose(),X)),X.T)) 

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        
        for num_epoch in range(num_epochs):
            for i in range(0,X.shape[1],batch_size):
                if (X.shape[1] > i+batch_size):
                    last = i + batch_size  
                else: 
                    last = X.shape[1] 
                actual = self.predict(X[:,i:last])
                expected = y[:,i:last]
                error = expected-actual
                out_fn = X[:,i:last].transpose()
                if learning.lower() == 'filtered':
                    self.weights = (1 - gamma)*self.weights + alpha*np.dot(expected,out_fn)
                elif learning.lower() == 'delta':
                    self.weights = self.weights + alpha*np.dot(error,out_fn)
                elif learning.lower() == 'unsupervised_hebb':
                    self.weights = self.weights + alpha*np.dot(actual,out_fn)

    def calculate_mean_squared_error(self, X, y):
        
        
        error = ((self.predict(X)-y)**2).mean()
        return error    
