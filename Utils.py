import numpy as np

def vector_to_matrices(vector, sizes):
    #print "LOL!", vector
    matrices = []
    start = 0
    for s in sizes:
        m = vector[start:start+(s[0]*s[1])]
        start += s[0]*s[1]
        matrices.append(m)
    return matrices

def matrices_to_vector(matrices):
    return np.hstack(matrices)