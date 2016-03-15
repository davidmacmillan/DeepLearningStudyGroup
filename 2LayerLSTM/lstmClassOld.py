import theano
import theano.tensor as T
import numpy as np
import random
import matplotlib.pyplot as plt
import cPickle as pickle
from math import sqrt
#from lstmClass import LstmLayer, recurrent_fn

'''Define lstm class for single layer lstm
The objective is to define in such a way as to facilitate construction of multi-layer
lstm.  The questions are:
1.  Cost function can't go inside the class because it may only be associated with the last
layer in the stack.
2.  Should the scan function be inside or outside of the class?
3.  How about the gradient calculations?  It seems like those need to be outside the class
does that mean that the gradient calculations have to be outside the class statement.
4.  Perhaps the scan function and the single layer recurrance function need to be inside
the class statement, but the cost function goes outside.
5.  Then the gradient calculation might only need to have a list of the parameters for which
the cost needs to be diff'd.  That would just be the list of lstm-layer objects dotted with
the parameter list for each one.
6.  Not clear how gradient of scan function may interact with python oop.  Not sure if scan
output includes enough for gradient calc.  Perhaps scan should be external to class structure
Plan A.
class RNN


'''

class LstmLayer(object):

    def __init__(self, n_in, n_hidden, n_out, name):
        self.name = name
        rng = np.random.RandomState(1234)
        #cell input
        self.W_ug = np.asarray(rng.normal(size=(n_in, n_hidden), scale= .01, loc = 0.0), dtype = theano.config.floatX)
        self.W_hg = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = 0.0), dtype = theano.config.floatX)
        self.b_g = np.zeros((n_hidden,), dtype=theano.config.floatX)
        #input gate equation
        self.W_ui = np.asarray(rng.normal(size=(n_in, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.W_hi = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.b_i = np.zeros((n_hidden,), dtype=theano.config.floatX)
        #forget gate equations
        self.W_uf = np.asarray(rng.normal(size=(n_in, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.W_hf = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.b_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        #cell output gate equations
        self.W_uo = np.asarray(rng.normal(size=(n_in, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.W_ho = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.b_o = np.zeros((n_hidden,), dtype=theano.config.floatX)
        #output layer
        self.W_hy = np.asarray(rng.normal(size=(n_hidden, n_out), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.b_hy = np.zeros((n_out,), dtype=theano.config.floatX)

        #cell input
        self.W_ug = theano.shared(self.W_ug, 'W_ug' + self.name)
        self.W_hg = theano.shared(self.W_hg, 'W_hg' + self.name)
        self.b_g = theano.shared(self.b_g, 'b_g' + self.name)
        #input gate equation
        self.W_ui = theano.shared(self.W_ui, 'W_ui' + self.name)
        self.W_hi = theano.shared(self.W_hi, 'W_hi' + self.name)
        self.b_i = theano.shared(self.b_i, 'b_i' + self.name)
        #forget gate equations
        self.W_uf = theano.shared(self.W_uf, 'W_uf' + self.name)
        self.W_hf = theano.shared(self.W_hf, 'W_hf' + self.name)
        self.b_f = theano.shared(self.b_f, 'b_f' + self.name)
        #cell output gate equations
        self.W_uo = theano.shared(self.W_uo, 'W_uo' + self.name)
        self.W_ho = theano.shared(self.W_ho, 'W_ho' + self.name)
        self.b_o = theano.shared(self.b_o, 'b_o' + self.name)
        #output layer
        self.W_hy = theano.shared(self.W_hy, 'W_hy' + self.name)
        self.b_hy = theano.shared(self.b_hy, 'b_hy' + self.name)

        self.h0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        self.s0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        self.argList = [self.W_ug, self.W_hg, self.b_g, self.W_ui, self.W_hi,
                        self.b_i, self.W_uf, self.W_hf, self.b_f, self.W_uo, self.W_ho, self.b_o, self.W_hy, self.b_hy]

def recurrent_fn(u_t, h_tm1, s_tm1, W_ug, W_hg, b_g, W_ui, W_hi, b_i, W_uf, W_hf, b_f,
                     W_uo, W_ho, b_o, W_hy, b_hy):
    g_t = T.tanh(T.dot(u_t, W_ug) + T.dot(h_tm1, W_hg) + b_g)
    i_t = T.nnet.sigmoid(T.dot(u_t, W_ui) + T.dot(h_tm1, W_hi) + b_i)
    f_t = T.nnet.sigmoid(T.dot(u_t, W_uf) + T.dot(h_tm1, W_hf) + b_f)
    o_t = T.nnet.sigmoid(T.dot(u_t, W_uo) + T.dot(h_tm1, W_ho) + b_o)
    s_t = g_t * i_t + s_tm1*f_t
    h_t = T.tanh(s_t)*o_t
    #h_t = self.activ(T.dot(h_tm1, W_hh) + T.dot(u_t, W_uh) + b_hh)
    return [h_t, s_t]

def fcn2(u_t, h_tm1, s_tm1,h_tm12, s_tm12, W_ug, W_hg, b_g, W_ui, W_hi, b_i, W_uf, W_hf, b_f,
                     W_uo, W_ho, b_o, W_hy, b_hy, W_ug2, W_hg2, b_g2, W_ui2, W_hi2, b_i2, W_uf2, W_hf2, b_f2,
                     W_uo2, W_ho2, b_o2, W_hy2, b_hy2):
    [h_t, s_t] = recurrent_fn(u_t, h_tm1, s_tm1, W_ug, W_hg, b_g, W_ui, W_hi, b_i, W_uf, W_hf, b_f,
                     W_uo, W_ho, b_o, W_hy, b_hy)
    o1 = T.dot(h_tm1, W_hy) + b_hy
    [h_t2, s_t2] = recurrent_fn(o1, h_tm12, s_tm12, W_ug2, W_hg2, b_g2, W_ui2, W_hi2, b_i2, W_uf2, W_hf2, b_f2,
                     W_uo2, W_ho2, b_o2, W_hy2, b_hy2)
    return [h_t, s_t, h_t2, s_t2]


#use lstmLayer class to define algebra of lstm and build stack and gradient calculation

#one layer lstm stack for stock price prediction
# u = T.matrix()
# t = T.scalar()
# l1 = LstmLayer(n_in=5, n_hidden=10, n_out=1, name='l1')

#theano.printing.debugprint([h0_tm1, u, W_hh, W_uh, W_hy, b_hh, b_hy], print_type=True)
#define
# [l1.h, l1.s], _ = theano.scan(recurrent_fn, sequences = u,
#                            outputs_info = [l1.h0_tm1, l1.s0_tm1],
#                            non_sequences = l1.argList)
# y = T.dot(l1.h[-1], l1.W_hy) + l1.b_hy
# cost = ((t - y)**2).mean(axis=0).sum()
# grad = T.grad(cost, l1.argList)
# lr = T.scalar()
# update = [(a, a-lr*b) for (a,b) in zip(l1.argList, grad)]
#
# train_step = theano.function([u, t, lr], cost,
#             on_unused_input='warn',
#             updates=update,
#             allow_input_downcast=True)

#two layer lstm stack for stock price prediction
u = T.matrix()
t = T.scalar()
o1 = T.matrix()
l1 = LstmLayer(n_in=5, n_hidden=10, n_out=10, name='l1')
l2 = LstmLayer(n_in=10, n_hidden=10, n_out=1, name='l2')
#theano.printing.debugprint([h0_tm1, u, W_hh, W_uh, W_hy, b_hh, b_hy], print_type=True)
#define
[l1.h, l1.s, l2.h, l2.s], _ = theano.scan(fcn2, sequences = u,
                           outputs_info = [l1.h0_tm1, l1.s0_tm1, l2.h0_tm1, l2.s0_tm1],
                           non_sequences = l1.argList + l2.argList)
#                          non_sequences = l1.argList + l2.argList, mode='DebugMode')



y = T.dot(l2.h[-1], l2.W_hy) + l2.b_hy
cost = ((t - y)**2).mean(axis=0).sum()
grad = T.grad(cost, l1.argList + l2.argList)
lr = T.scalar()
update = [(a, a-lr*b) for (a,b) in zip(l1.argList + l2.argList, grad)]

train_step = theano.function([u, t, lr], cost,
            on_unused_input='warn',
            updates=update,
            allow_input_downcast=True)
#           allow_input_downcast=True, mode='DebugMode')


if __name__ == '__main__':

    (xlist, ylist) = pickle.load(open('stockTT.bin', 'rb'))
    nInputs = len(xlist[0])
    x = np.array(xlist, dtype = theano.config.floatX)
    y = np.array(ylist, dtype = theano.config.floatX)
    print "Std Dev of Price Change", np.std(y)
    nHidden = 20
    nOutputs = 1
    lr = 0.01
    eSmooth = 1.0
    nPasses = 1
    vals = []
    errSq = []
    for i in range(nPasses):
        for j in range(len(x)):
            u = np.asarray(xlist[j], dtype = theano.config.floatX).reshape((1,nInputs))
            t = y[j]

            c = train_step(u, t, lr)
            if j%10==0: print "iteration {0}: {1}".format(j, np.sqrt(c))
            eSmooth = 0.1*np.sqrt(c) + 0.9*eSmooth
            vals.append(eSmooth)
            errSq.append(c)
    print 'RMS Pred Error', sqrt(np.average(errSq[500:]))
    plt.plot(vals)
    plt.show()



