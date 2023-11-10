import numpy as np
import random

def sigmoid(z):
    '''Sigmoid function'''
    return 1./1.+np.exp(-z)

def sigmoid_prime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z)*(1-sigmoid(z))

'''
The parameter 'sizes' to create the artificial neural network (ANN)
is a list containing the number of neurons in each layer.
'''

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, input):
        '''Returns the output of the network for the given input'''
        for bias, weight in zip(self.biases, self.weights):
            input = sigmoid(np.dot(weight, input) + bias)
        return input
    
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        '''
        Train the ANN using mini-batch stochastic gradient descent
        '''
        if test_data: n_test = len(test_data)
        
        n = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, learning_rate):
        '''
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch.
        '''

        nabla_bias = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weight = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in mini_batch:
            delta_nabla_bias, delta_nabla_weight = self.backpropagation(x, y)
            nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias, delta_nabla_bias)]
            nabla_weight = [nw + dnw for nw, dnw in zip(nabla_weight, delta_nabla_weight)]

        self.weights = [w - (learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_weight)]
        self.biases = [b - (learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_bias)]
    
    def backpropagation(self, x, y):
        '''
        Returns a tuple (nabla_bias, nabla_weight) representing the gradient
        for the cost function C_x.
        '''
        nabla_bias = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weight = [np.zeros(weight.shape) for weight in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for bias, weight in zip(self.biases, self.weights):
             z = np.dot(weight, activation) + bias
             zs.append(z)
             activation = sigmoid(z)
             activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_bias[-1] = delta
        nabla_weight[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta)*sp
            nabla_bias[-layer] = delta
            nabla_weight[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (nabla_bias, nabla_weight)
    
    def evaluate(self, test_data):
        '''
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        '''

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)






a = Network([3,5,5,1])






