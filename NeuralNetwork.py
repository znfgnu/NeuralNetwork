#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import Utils as ut
import scipy.optimize as opt

sgm = lambda x: 1. / (1. + np.exp(-x))

class NeuralLayer:
    def __init__(self, ins, outs):
        # ins - without bias
        self._size = (ins+1, outs)
        self._theta = np.random.random((ins+1)*outs)

    def run(self, inp):
        self._lastInput = inp
        self._lastZ = (np.hstack([inp, [1]]) * np.matrix(self._theta).reshape(self._size)).getA1()
        self._lastOutput = sgm(self._lastZ)
        return self._lastOutput

class NeuralNetwork:
    def __init__(self, size):
        self._mylambda = .01
        self._layers = []
        for ins, outs in zip(size[:-1], size[1:]):
            l = NeuralLayer(ins, outs)
            self._layers.append(l)


    def run(self, input):
        last_out = np.array(input)
        for l in self._layers:
            last_out = l.run(last_out)
        return last_out

    def cost(self, trainset, thetaVec=None):
        if thetaVec != None:
            #apply thetaVec
            sizes = [l._size for l in self._layers]
            newThetas = ut.vector_to_matrices(thetaVec, sizes)
            for l, t in zip(self._layers, newThetas):
                #print "!!!", t, "!!!"
                l._theta = t

        c = lambda (y, h): -y*np.log(h) -(1. - y)*np.log(1.-h)

        total_cost = 0.
        for x, y in trainset:
            yarr = np.array(y)
            res = self.run(x)
            total_cost += sum( map(c, zip(yarr, res)) )

        total_cost /= len(trainset)

        lambda_part = 0.
        for l in self._layers:
            lambda_part = np.multiply(l._theta, l._theta).sum()

        lambda_part *= self._mylambda / (2.*len(trainset))

        total_cost += lambda_part

        return total_cost

    def train(self, trainset):
        c = lambda t : self.cost(trainset, t)
        initial_theta = ut.matrices_to_vector([l._theta for l in self._layers])
        #print initial_theta
        optimum_theta = opt.fmin_bfgs(c, initial_theta)


def main():
    trainset = [
        ([0,0],[0]),
        ([0,1],[1]),
        ([1,0],[1]),
        ([1,1],[0]),
    ]
    nn = NeuralNetwork((2,2,1))

    for x, y in trainset:
        print nn.run(x)

    nn.train(trainset)

    for x, y in trainset:
        print nn.run(x)

if __name__ == "__main__":
    main()