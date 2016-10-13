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

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    x_dim = 50
    concat_len = x_dim + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5,0.2,0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(100):
        print "cur iter: ", cur_iter
        print "input_val_arr=", input_val_arr
        print "y_list=", y_list
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0])

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print "loss: ", loss
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()

