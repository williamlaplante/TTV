import torch as tc
import numpy as np
from torch import nn
from torch.distributions.constraint_registry import transform_to
from distributions import Normal
from torch.autograd import Variable

def init_weights(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/n
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0.0)

class LSTM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LSTM, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = 5
        self.hidden_size = 30
 
        self.lstm = nn.LSTM(input_size=dim_in, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True) #lstm
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, dim_out)
    
    def forward(self,x):
        x = x.view(-1, *x.shape)
        h_0 = Variable(tc.zeros(self.num_layers, self.hidden_size)) #hidden state
        c_0 = Variable(tc.zeros(self.num_layers, self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.fc1(hn)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)

        return out.reshape(-1)

#Typical NN. Takes input dimension (conditional), output dimension (# conditional dist parameters)
class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NeuralNetwork, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        #self.hidden_dim = int((dim_in + dim_out)/2)
        #self.num_hidden_layers = 4
        self.activation = nn.Tanh()
        self.flatten = nn.Flatten()

        #building the network
        """
        self.layers = [nn.Linear(dim_in, self.hidden_dim), self.activation]
        for _ in range(self.num_hidden_layers) : self.layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), self.activation])
        self.layers.append(nn.Linear(self.hidden_dim, self.dim_out))
        
        self.net = nn.Sequential(*self.layers)"""


        
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, dim_out),
        )
        
        self.num_params = 0
        #self.net.apply(init_weights) #initialize weights

    def forward(self, x):
        return self.net(x)

#The proposal, P(X | Y) = P(x_1 | Y, x_2,...x_n)P(x_2 | Y, x_3, ..., x_n)...P(x_n | Y)
#having the inverse dependencies allows us the remove some of the conditioning (and thus less parameter estimations). But it's not mandatory.
class Proposal():
    def __init__(self, g, lstm):
        super().__init__()
        self.lstm = lstm
        self.graph = g
        self.reverse_ordered_vars = g.reverse_topological()
        self.reverse_ordered_latent_vars = [var for var in self.reverse_ordered_vars if var not in g.Graph["Y"].keys()]
        self.n = len(g.Graph["Y"]) #number of observations

        self.distributions = {var : Normal for var in self.reverse_ordered_latent_vars}
        self.constraints = {var : self.distributions[var].arg_constraints for var in self.reverse_ordered_latent_vars}
        
        if self.lstm:
            self.links = {var : LSTM(dim_in = self.n + i, dim_out = self.distributions[var].NUM_PARAMS) for i, var in enumerate(self.reverse_ordered_latent_vars)}
        else:
            self.links = {var : NeuralNetwork(dim_in = self.n + i, dim_out = self.distributions[var].NUM_PARAMS) for i, var in enumerate(self.reverse_ordered_latent_vars)}
        
        return
    
    def sample(self, y : tc.Tensor) -> dict:
        _sample = {}
        val = y #start by conditioning on y
        with tc.no_grad():
            for latent in self.reverse_ordered_latent_vars:
                #get output of NN. Should be of dimension == # dist params
                nn_out = self.links[latent].forward(val)
                #iterate over nn output and the associated support for each element (ex : mu, sigma for a Normal, hence sigma has (0,inf) support) and apply appropriate transform
                dist_params = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)] #apply constraint transform
                #sample from the conditional distribution
                s = self.distributions[latent](*dist_params).sample()
                _sample.update({latent : s})
                if s.ndim == 0 : s = s.reshape(-1)
                val = tc.cat([y, s])

        return _sample
    
    def log_prob(self, sample : tc.Tensor, y : tc.Tensor) -> tc.Tensor:
        lik = tc.tensor(0.0)
        val = y
        #we need to flip sample, since it comes in topological order and we iterate over the reverse order.
        for latent, x in zip(self.reverse_ordered_latent_vars, tc.flip(sample, [0])):
            nn_out = self.links[latent].forward(val)
            dist_params = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)]
            lik += self.distributions[latent](*dist_params).log_prob(x)
            if x.ndim == 0: x = x.reshape(-1)
            val = tc.cat([val, x])

        return lik
    
    def get_params(self):
        params = []
        num_params = 0
        for latent, neuralnet in self.links.items():
            params += [*neuralnet.parameters()]
            num_params += sum(p.numel() for p in neuralnet.parameters() if p.requires_grad)
        self.num_params = num_params
        return params

