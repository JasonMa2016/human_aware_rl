import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Normal 

from collections import OrderedDict 
from maml_rl.policies.policy import Policy, weight_init 


class NormalMLPPolicy(Policy):
    """
    Policy network based on a multi-layer perceptron (MLP), with a 
    'Normal' distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces.
    """
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 activation=F.tanh, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size,
                                              output_size=output_size)

        self.hidden_sizes = hidden_sizes # hidden layers. eg. (100, 100)), two-layer 100-unit
        self.activation = activation # non-linear activation layer
        self.min_log_std = math.log(min_std)
        self.layer_sizes = (input_size,) + hidden_sizes + (output_size,) # include input and output layer 

        # create hidden layers
        for i in range(1, len(self.layer_sizes)):
            self.add_module('layer{}'.format(i),
                nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i]))

        self.sigma = nn.Parameter(torch.Tensor(output_size)) # why output size?
        self.sigma.data.fill_(math.log(init_std)) # what's the logic behind this?
        self.apply(weight_init) # initialize all layers

    def forward(self, input, params=None):
        # use current policy if no other parameters provided
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input 

        # pass through hidden layers 
        for i in range(1, len(self.layer_sizes)):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                                        bias=params['layer{0}.bias'.format(i)])
            # non-linear activation except in the last layer
            if i != len(self.layer_sizes) - 1:
                output = self.activation(output)
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        
        return Normal(loc=output, scale=scale)
