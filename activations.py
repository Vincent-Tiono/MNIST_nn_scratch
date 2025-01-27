import numpy as np
from math import exp

class ReLU:
    '''
    f(x) = {0 x<=0
            x x>0}
    '''
    
    def __call__(self, pre_activated_output):
        return np.maximum(0, pre_activated_output)
    
    def derivative(self, pre_activated_output, grad_so_far):
        return np.where(pre_activated_output <= 0, 0, 1) * grad_so_far
     
class Sigmoid:
    '''
    f(x) = 1 / (1 + e^(-x))
    '''
    
    def __call__(self, pre_activated_output):
        pre_activated_output = np.clip(pre_activated_output, -1000, 1000)
        return 1 / (1 + np.exp(-pre_activated_output))
    
    def derivative(self, pre_activated_output, grad_so_far):
        pre_activated_output = np.clip(pre_activated_output, 1000, -1000)
        return 1 / (1 + np.exp(-pre_activated_output)) * (1 - 1 / (1 + np.exp(-pre_activated_output))) * grad_so_far

class Softmax:
    '''
    pre_activated_output = [5, 2, -3]
    
    Softmax ->
    
    pre_activated_output_shifted = [0, -3, -8]
    
    exp_shifted = [e^0, e^-3, e^-8]
    denominator = e^0 + e^-3 + e^-8
    return [e^0 / denominator, e^-3 / denominator, e^-8 / denominator]
    '''
    def __call__(self, pre_activated_output):
        exp_shifted = np.exp(pre_activated_output - np.max(pre_activated_output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator
    
    def derivative(self, pre_activated_output, grad_so_far):
        output = self(pre_activated_output) # Get activated outputs for formulae
        batch_size, n_classes = output.shape
        
        # For 1 example, the jacobian is of size NxN, so for B batches, it is BxNxN
        jacobian = np.zeros((batch_size, n_classes, n_classes))
        
        for b in range(batch_size):
            out = output[b].reshape(-1, 1) # Flatten output to be an Nx1 matrix
            jacobian[b] = np.diagflat(out) - np.dot(out, out.T) # Create Jacobian for particular example
            
        return np.einsum('bij,bj->bi', jacobian, grad_so_far) # Efficient batch-wise dot product using Einstein summation notation
            