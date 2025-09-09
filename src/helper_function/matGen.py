import numpy as np
from scipy.linalg import orth
from scipy.linalg import svd
import torch

def genRandMatrix(numRows, numCols):
    # generate random orthonormal matrix to initialize as random weights
    randMat = np.random.normal(0, 1, size=(numRows, numCols))
    # print("generated shape: ", randMat.shape)
    U, s, V = torch.svd(torch.tensor(randMat.reshape(numRows, numCols)))
    # print(" U shape: ", U.shape)
    # print("V shape : ", V.shape)
    return V

def genPCAMat(samples):
    """
    Generate 
    """
    numRows, numCols = samples.shape
    U, s, V = svd(torch.tensor(samples.reshape(numRows, numCols)))
    return U

def genScaledRandMat(numRows, numCols, norm=None):
    if norm is None:
        norm = numRows + numCols
    orthMat = genRandMatrix(numRows, numCols)
    # norm_dim = min(numCols, numRows)
    if numRows <= numCols:
        coefs = torch.rand(size=(numRows, 1))
    else:
        coefs = torch.rand(size=(1, numCols))
        
    coefs_sum = sum(coefs)
    # print("coefficients: ", coefs.shape)
    normed_coefs = coefs.T * (norm/coefs_sum)
    return normed_coefs * orthMat