from __future__ import print_function

import numpy as np
from numpy import array, dot, sum, random, where, argmax, absolute
from math import sqrt
from pandas import DataFrame

from pprint import pprint
from numpy import array, where, vectorize, exp, log


def modify(line):
    return map(int, line[:-1]), line[-1]

def result_to_vector(yi):
    if yi == 'win': 
        return ([1, 0, 0])
    elif yi == 'loss':
        return ([0, 0, 1])
    else:
        return ([0, 1, 0])

training_file = './data/connect_4/train.data' 
test_file = './data/connect_4/test.data'

with open(training_file) as f:
    tdata = [modify(line.strip().split(',')) for idx, line in enumerate(f)]
x, y = map(list, zip(*tdata))
xtrain, ytrain = array(x), array(map(result_to_vector, y))

with open(test_file) as f:
    tdata = [modify(line.strip().split(',')) for idx, line in enumerate(f)]
x, y = map(list, zip(*tdata))
xtest, ytest = array(x), array(map(result_to_vector, y))

def get_training_xy(train=True):
    return [(xtest, ytest), (xtrain, ytrain)][train]

def signum(x): 
    return where(x >= 0, 1.0, 0.0)

def sigmoid(x):
    return 1 / (1 + exp(-x))

def softplus(x):
    return log(1 + exp(x))


# settings 
MODE = "SIGMOID"
nonlinear = sigmoid if MODE == "SIGMOID" else softplus
PASSES = 50
HIDDEN_LAYER_SIZE = 100
FIXED_UPDATE_RULE = True
SEED = 1 
random.seed(SEED)
axis = ['win', 'draw', 'loss']


def accuracy(hidden_layer, output_layer, train=True):
    x, y = get_training_xy(train=train)
    correct_cnt, total = 0, 0
    conf_matr = {axis[i] : \
                    {axis[j]: 0 for j in [0, 1, 2]} \
                                    for i in [0, 1, 2]}
    for xi, yi in zip(x, y):
        # forward pass for prediction
        xi_in = xi.reshape(1, *xi.shape).transpose()
        hidden_layer.forward_pass(xi_in)
        output_layer.forward_pass(hidden_layer.o)
        # calculating prediction
        output = output_layer.o.transpose()
        target = array([yi])[0]
        predict = argmax(output)
        correct = argmax(target)
        # verifying prediction
        if predict == correct:
            correct_cnt += 1
        total += 1
        conf_matr[axis[correct]][axis[predict]] += 1
    accuracy = float(correct_cnt) / float(total)
    print('\tcorrect = {}'.format(correct_cnt))
    print('\ttotal = {}'.format(total))
    print('\taccuracy = {}'.format(accuracy))
    # print DataFrame(conf_matr)
    print('- ' * 10)
    return accuracy


class Layer(object):
    def __init__(self, n_prev, n_next, hidden):
        self.n_prev = n_prev
        self.n_next = n_next
        self.hidden = hidden
        self.w = random.rand(n_prev * n_next).reshape(n_next, n_prev)-0.5
        if MODE == "SOFTPLUS":
            self.w *= 0.01
        self.learning_rate = 0.1  

    def forward_pass(self, x):
        self.x = np.append([[1.0]], x, axis=0)
        self.net = dot(self.w, self.x)
        self.o = nonlinear(self.net)
        return self.o

    def reverse_pass(self, e):
        if MODE == "SIGMOID":
            self.delta = e * self.o * (1.0 - self.o)
        else:
            self.delta = e * sigmoid(self.net)
        update = dot(self.delta, self.x.transpose())
        # print ('delta = ', self.delta)
        # print ('weights = ', self.w)
        if not self.hidden:
            dE_do_hidden = dot(self.w.transpose(), self.delta)
        self.w -= (self.learning_rate * update)
        if not self.hidden:
            return dE_do_hidden


def backpropogation(x, y, hidden_layer, output_layer):
    strikes = 0; prev_acc = 0.0
    for passthrough in xrange(PASSES):
        print ('pass through = ', passthrough, ' ====>')
        for xi, yi in zip(x, y):
            # forward pass through hidden layer 
            xi_in = xi.reshape(1, *xi.shape).transpose()
            hidden_layer.forward_pass(xi_in)
            # forward pass though the output layer
            output_layer.forward_pass(hidden_layer.o)
            # reverse pass through the output layer
            target = array([yi]).transpose()
            dE_do_output = output_layer.o - target
            dE_do_hidden = output_layer.reverse_pass(dE_do_output)
            # reverse pass through the hidden layer
            hidden_layer.reverse_pass(dE_do_hidden[:-1])
        accuracy(hidden_layer, output_layer, train=True) # training
        acc = accuracy(hidden_layer, output_layer, train=False) # test 
        diff = abs(prev_acc - acc)
        prev_acc = acc
        if diff < 0.002:
            strikes += 1
            if(strikes == 4): 
                break
        else:
            strikes = 0
        print('strikes = {}'.format(strikes))
        print('diff = {}'.format(diff))
        print('-' * 50)
        if not FIXED_UPDATE_RULE:
            hidden_layer.learning_rate = 0.1 / sqrt(passthrough + 2)
            output_layer.learning_rate = 0.1 / sqrt(passthrough + 2)

if __name__ == '__main__':
    # read the training data
    print('-' * 50)
    print('HIDDEN_LAYER_SIZE = {}, PASSES = {}'.format(HIDDEN_LAYER_SIZE, PASSES))
    print('SEED = {}'.format(SEED))
    print('UPDATE_RULE = {}'.format(str(FIXED_UPDATE_RULE)))
    print('-' * 50)
    x, y = get_training_xy()
    # build the network
    hidden_layer = Layer(127, HIDDEN_LAYER_SIZE, hidden=True)
    output_layer = Layer(HIDDEN_LAYER_SIZE + 1, 3, hidden=False)
    # train
    print('before ==>')
    accuracy(hidden_layer, output_layer, train=True)
    accuracy(hidden_layer, output_layer, train=False)
    print('backpropgation ===> ')
    backpropogation(x, y, hidden_layer, output_layer)
