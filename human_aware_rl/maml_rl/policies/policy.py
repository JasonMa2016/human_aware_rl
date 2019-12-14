import torch 
import torch.nn as nn 

from collections import OrderedDict

def weight_init(module):
    """
    Initialize a linear layer weights.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight) # set weights uniformly according to Glorot & Bengio, 2010.
        module.bias.data.zero_() # set bias to 0
    return module

class Policy(nn.Module):
    """
    Base policy class.
    """
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.05, algorithm='MAML'):
        """
        Apply one step of gradient descent on the loss function 'loss', with
        step_size 'step_size', and returns the updated parameters of the 
        neural network.
        """

        if algorithm == 'ANIL':
            # compute gradients with respect to the last layer
            grads = torch.autograd.grad(loss, list(self.parameters())[-2:],
             create_graph= True)
            updated_params = OrderedDict(list(self.named_parameters())[:-2])
            for (name,param), grad in zip(list(self.named_parameters())[-2:], grads):
                updated_params[name] = param - step_size * grad 
        else:
            # determine if first-order approximation of the gradient
            first_order = True if algorithm == 'FOMAML' else False
            # compute gradients
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=not first_order)
            updated_params = OrderedDict()
            # apply gradients
            for (name, param), grad in zip(self.named_parameters(), grads):
                updated_params[name] = param - step_size * grad

        return updated_params, grads