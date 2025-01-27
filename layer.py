import numpy as np
from activations import *
from random import shuffle

# Represents a single layer of the neural network 
class Layer:
    def __init__(self, input_size, output_size, bias=False, activation_func=None):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = 1e-3
        self.bias = bias
        self.activation_func = activation_func
        
        self.weights = np.random.rand(self.input_size, self.output_size)
        
        if bias:
            self.weights = np.vstack((self.weights, np.ones((1, self.weights.shape[1]))))
        
        self.pre_activated_output = None
        self.current_inputs = None
        self.update_matrix = None
        #print(self.weights)
        
    #def get_batch_size(self):
    #    return self.batch_size
    
    def set_alpha(self, new_alpha):
        self.alpha = new_alpha
        
    def __call__(self, layer_inputs):
        '''
        layer_inputs = (batch_size, input_size +1)
        weights = (input_size +1, output_size)
        '''
        if self.bias:
            layer_inputs = np.hstack((layer_inputs, np.ones((input.shape[0], 1))))
            
        self.current_inputs = np.copy(layer_inputs)
            
        layer_outputs = layer_inputs @ self.weights
        
        #print(layer_outputs)
        
        if self.activation_func:
            self.pre_activated_output = np.copy(layer_outputs)
            layer_outputs = self.activation_func(layer_outputs)
            
        return layer_outputs
    
    def back(self, ret):
        if self.activation_func:
            ret = self.activation_func.derivative(self.pre_activated_output, ret)
        
        self.update_matrix = self.current_inputs.T @ ret
        new_ret = ret @ self.weights.T
        
        if self.bias:
            new_ret = new_ret[:, :-1]
        
        return new_ret
            
    def update(self):
        self.weights -= self.alpha * self.update_matrix
        self.pre_activated_output = None
        self.current_inputs = None
        self.update_matrix = None
    
# Represents the enitre neural network
class LayerList:
    def __init__(self, *layers):
        if len(layers) == 0:
            self.model = list()
        else:
            self.model = list(layers)
        
    def append(self, *layers):
        for layer in layers:
            self.model.append(layer)
            
    def set_alpha(self, new_alpha):
        for layer in self.model:
            layer.set_alpha(new_alpha)
            
    def __call__(self, model_input):
        for layer in self.model:
            # Call __call__ of Layer class
            model_input = layer(model_input)
        return model_input
    
    def back(self, error):
        for layer in self.model[::-1]:
            error = layer.back(error)
    
    def step(self): 
        for layer in self.model:
            layer.update()

    def predict(self, inputs):
        predictions = []
        
        for inp in inputs:
            
            predictions.append(self(np.expand_dims(inp, axis=0)))
            
        return predictions

    @staticmethod
    # Create batches of input data and expected
    def batch(input_data, expected, batch_size):
        num_data = input_data.shape[0]
        indices = [i for i in range(num_data)]
        shuffle(indices)
        
        batched_input, batched_expected = [], []
        
        for i in range(num_data // batch_size):
            batch_inp, batch_exp = [], []
            
            for j in range(batch_size):
                batch_inp.append(input_data[i * batch_size + j])
                batch_exp.append(expected[i * batch_size + j])
                #batch_inp.append(input_data[indices[i * batch_size + j]])
                #batch_exp.append(expected[indices[i * batch_size + j]])
                
            batched_input.append(np.array(batch_inp))
            batched_expected.append(np.array(batch_exp))
            
        return np.array(batched_input), np.array(batched_expected)
        
    '''
    def fit(self, input_data, expected, epochs, alpha, batch_size, loss_deriv_func):
        self.set_alpha(alpha)
        
        prev_update = 1
        
        for e in range(epochs):
            batched_input, batched_expected = LayerList.batch(input_data, expected, batch_size)
            
            for i in range(len(batched_input)):
                model_output = self(batched_input[i])
                self.back(loss_deriv_func(model_output, batched_expected[i]))
                self.step()
            
            if e == 10 * prev_update:
                alpha /= 10
                self.set_alpha(alpha)
                prev_update = e
    '''
    
    def fit(self, input_data, expected, epochs, alpha, batch_size, loss_deriv_func):
        """Model training loop

        Args:
            input_data (np.array): model training data
            expected (np.array): expected values for training data
            epochs (int): number of times the input_data is fed to the model
            alpha (float): initial learning rate
            batch_size (int): batch size for training
            loss_deriv_func (function): loss function (from loss.py)
        """
        if len(self.model) == 0:
            return
        
        total_iter = epochs
        self.set_alpha(alpha)

        while epochs:
            epochs -= 1
            batched_data, batched_expected = LayerList.batch(input_data, expected, batch_size)
            
            for idx, data_batch in enumerate(batched_data):
                output = self(data_batch)
                self.back(loss_deriv_func(output, batched_expected[idx]))
                self.step()
            
            if epochs == total_iter // 10:
                # Reducing learning rate to hone in on minima of loss function
                alpha /= 10
                self.set_alpha(alpha)        
            
    


if __name__ == "__main__":
    model = LayerList(Layer(1, 1, 0.1, 1), Layer(1, 2, 0.1, 1, activation_func=Softmax()))
    inp = np.array([[1]])
    print(model(inp))
    