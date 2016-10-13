import numpy as np
import sys

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

class Primes:
    def __init__(self):
        self.primes = list()
        for i in range(2, 100):
            is_prime = True
            for j in range(2, i-1):
                if i % j == 0:
                    is_prime = False
            if is_prime:
                self.primes.append(i)
        self.primes_count = len(self.primes)
    def get_sample(self, x_dim, y_dim, index):
        result = np.zeros((x_dim+y_dim))
        for i in range(index, index + x_dim + y_dim):
            result[i-index] = self.primes[i%self.primes_count]/100.0
        return result


def example_0():
    mem_cell_ct = 100
    x_dim = 50
    concat_len = x_dim + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork(lstm_param)

    primes = Primes()
    x_list = []
    y_list = []
    for i in range(0, 10):
        sample = primes.get_sample(x_dim, 1, i)
        x = sample[0:x_dim]
        y = sample[x_dim:x_dim+1].tolist()[0]
        x_list.append(x)
        y_list.append(y)

    for cur_iter in range(10000):
        if cur_iter % 1000 == 0:
            print "y_list=", y_list
        for ind in range(len(y_list)):
            lstm_net.x_list_add(x_list[ind])
            if cur_iter % 1000 == 0:
                print "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0])

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        if cur_iter % 1000 == 0:
            print "loss: ", loss
        lstm_param.apply_diff(lr=0.01)
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()
