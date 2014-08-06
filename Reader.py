__author__ = 'konrad'

import NeuralNetwork as nn
import csv
import time

def main():
    n = nn.NeuralNetwork((5,10,10,4))

    trainset = []
    testset = []

    d = {
        'very_low':[1,0,0,0],
        'Very Low':[1,0,0,0],
        'Low':[0,1,0,0],
        'Middle':[0,0,1,0],
        'High':[0,0,0,1],
         }

    with open("training_data.csv") as f:
        rdr = csv.reader(f)
        for line in rdr:
            out = d[line[-1]]
            inp = map(float, map(lambda s: s.replace(',','.'),line[:-1]))
            trainset.append((inp,out))
    now = time.time()
    n.train(trainset)
    after = time.time()
    print "Time:", after-now

    with open("test_data.csv") as f:
        success = 0
        total = 0
        rdr = csv.reader(f)
        for line in rdr:
            total+=1
            print line
            out = d[line[-1]]
            inp = map(float, map(lambda s: s.replace(',','.'),line[:-1]))
            res = [int(x.round()) for x in n.run(inp)]
            if res==out: success+=1
            else: print res, "instead of", out
            print success, "of", total

if __name__=="__main__":
    main()