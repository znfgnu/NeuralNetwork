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
        eps = 1e-2
        self._theta = np.random.random((ins+1)*outs)

    def run(self, inp):
        self._lastInput = np.hstack(([1],inp))
        self._lastZ = self._lastInput.dot(self._theta.reshape(self._size))
        self._lastOutput = sgm(self._lastZ)
        return self._lastOutput

    def clearAcc(self):
        self._capitalDelta = np.zeros(self._size[0]*self._size[1])

    def compute_delta(self, next_delta):
        self._delta = np.multiply(next_delta[1:].dot(self._theta.reshape(self._size).transpose()),
                                  np.multiply(self._lastInput, (1-self._lastInput)))
        # ACHTUNG!
        tmp = self._lastInput.reshape((self._size[0],1))
        tmp2 = next_delta[1:].reshape((1,self._size[1]))
        self._capitalDelta += tmp.dot(tmp2).flatten()
        return self._delta
        pass

class NeuralNetwork:
    def __init__(self, size):
        self._mylambda = .00001
        self._layers = []
        for ins, outs in zip(size[:-1], size[1:]):
            l = NeuralLayer(ins, outs)
            self._layers.append(l)
        self._lastThetaVec=None


    def run(self, input):
        last_out = np.array(input)
        self._lastThetaVec = last_out
        for l in self._layers:
            last_out = l.run(last_out)
        return last_out

    def _backprop(self, result, expected):
        result_delta = result - expected
        last_delta = np.hstack(([1],result_delta)) #?
        for l in reversed(self._layers):
            last_delta = l.compute_delta(last_delta)

    def _derivatives(self, trainset, thetaVec=None):
        if thetaVec!=None and thetaVec!=self._lastThetaVec:
            self.cost(trainset, thetaVec)

        res = ut.matrices_to_vector([
            (l._capitalDelta / len(trainset)) + (np.hstack((np.zeros(l._size[1]), l._theta[l._size[1]:] * self._mylambda / len(trainset))))# / len(trainset)# + l._theta * self._mylambda * <333>.ravel()
            for l in self._layers])

        return res

    def cost(self, trainset, thetaVec=None):
        if thetaVec != None:
            #apply thetaVec
            sizes = [l._size for l in self._layers]
            newThetas = ut.vector_to_matrices(thetaVec, sizes)
            for l, t in zip(self._layers, newThetas):
                #print "!!!", t, "!!!"
                l._theta = t

        for l in self._layers: l.clearAcc()

        c = lambda (y, h): -y*np.log(h) -(1. - y)*np.log(1.-h)

        total_cost = 0.
        for x, y in trainset:
            yarr = np.array(y)
            res = self.run(x)
            total_cost += sum( map(c, zip(yarr, res)) )
            self._backprop(res, yarr)

        total_cost /= len(trainset)

        lambda_part = 0.
        for l in self._layers:
            lambda_part += np.multiply(l._theta[l._size[1]:], l._theta[l._size[1]:]).sum()

        lambda_part *= (self._mylambda / (2.*len(trainset)))

        total_cost += lambda_part

        return total_cost

    def train(self, trainset):
        c = lambda t : self.cost(trainset, t)
        d = lambda t : self._derivatives(trainset, t)
        initial_theta = ut.matrices_to_vector([l._theta for l in self._layers])
        c(initial_theta)
        optimum_theta = opt.fmin_bfgs(c, initial_theta, fprime=d, disp=False)
        print optimum_theta

def main():
    trainset = [
        ([0,0],[0]),
        ([0,1],[1]),
        ([1,0],[1]),
        ([1,1],[0]),
    ]

    tries = 20
    success = 0

    for i in range(tries):
        print success, "of", i, "tries"
        #for x, y in trainset:
        #    print nn.run(x)

        nn = NeuralNetwork((2,3,1))
        nn.train(trainset)
        #for x, y in trainset:
        #    res = nn.run(x)
        #    print y[0], int(res.round()), res

        a =  [y[0] for x,y in trainset]
        c = [nn.run(x) for x,y in trainset]
        b = [int(x.round()) for x in c]

        print "a:", a
        print "b:", b
        print "c:", c
        print
        if a==b: success+=1

    print success, "of", tries, "tries."

if __name__ == "__main__":
    main()