import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-0.1*x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def cross_entropy(x,y):
    if y == 1:
      return -np.log(x)
    else:
      return -np.log(1 - x)

def cross_entropy_prime(x,y):
        return -y / x + (1-y)/(1-x)

def keras_model(X, y):
    # create model
    model = Sequential()
    model.add(Dense(3, input_dim = 2, activation = 'sigmoid'))
    model.add(Dense(3, activation = 'sigmoid'))
    model.add(Dense(3, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))
    # compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # fit the model
    model.fit(X, y, epochs = 500, batch_size = 1)
    # evaluate the model
    scores = model.evaluate(X, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print(model.get_weights())

class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        self.learning_rate = 0.1
        self.epochs = 500000

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(self.epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            #error = -cross_entropy_prime(a[-1],y[i])/100.0
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += self.learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: 
                print ('epochs:', k)
                self.validate(np.delete(X,0,axis=1),y)

    def predict(self, x): 
        a = np.concatenate((np.array([[1]]), np.array([x])), axis=1)    
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_parameter(self):
        print("weights:",self.weights)
        print("hidden roles:","np.random.randint(X.shape[0])")
        print("slope:",0.1)
        print("learning_rate:",self.learning_rate)
        print("epochs:",self.epochs)

    def validate(self,X,y):
        sum = 0
        loss_sum = 0
        for i in range(X.shape[0]):
            pre = self.predict(X[i])
            loss_sum = loss_sum + cross_entropy(pre,y[i])
            if np.abs(y[i]-pre) < 0.5:
                sum = sum + 1
        print("acc:",sum/X.shape[0])
        print("loss:",loss_sum/X.shape[0],"\n")

if __name__ == '__main__':

    nn = NeuralNetwork([2,2,2,1])
    # X = np.array([[0, 0],
    #               [0, 1],
    #               [1, 0],
    #               [1, 1]])
    # y = np.array([0, 1, 1, 0])

    X = []
    for i in range(24):
        x1 = np.random.uniform(1,10)
        x2 = np.random.uniform(-1,1)
        X.append([x1,x2])
    X.append([1,0])
    for i in range(24):
        x1 = np.random.uniform(-10,-1)
        x2 = np.random.uniform(-1,1)
        X.append([x1,x2])
    X.append([-1,0])
    for i in range(24):
        x1 = np.random.uniform(-1,1)
        x2 = np.random.uniform(1,10)
        X.append([x1,x2])
    X.append([0,1])
    for i in range(24):
        x1 = np.random.uniform(-1,1)
        x2 = np.random.uniform(-10,-1)
        X.append([x1,x2])
    X.append([0,-1])

    X = np.array(X)
    ones = np.ones((50,), dtype=int).T
    zeros = np.zeros((50,), dtype=int).T
    y = np.concatenate((zeros,ones), axis=0)

    nn.fit(X, y)
    for e in X:
        print(e,nn.predict(e))
    nn.print_parameter()
    k_nn = keras_model(X,y)