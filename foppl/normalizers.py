import torch as tc
import numpy as np
from torch import nn
from torch.distributions.constraint_registry import transform_to
from utils import softplus, inverse_softplus

def init_weights(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/n
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0.0)



class Normal(tc.distributions.Normal):
    NUM_PARAMS = 2
    def __init__(self, loc, scale): # Define an optimizable scale that exists on the entire real line
        self.optim_scale = inverse_softplus(scale)
        super().__init__(loc, scale)

    def params(self): # Return a list of standard parameters of the distribution
        return [self.loc, self.scale]

    def optim_params(self): # Return a list of (optimizeable) parameters of the distribution
        return [self.loc.requires_grad_(), self.optim_scale.requires_grad_()]

    def log_prob(self, x): # This overwrites the default log_prob function and updates scale
        self.scale = softplus(self.optim_scale) # Needed to carry the gradient through
        return super().log_prob(x)


class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NeuralNetwork, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
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

#A normalizer is a proposal of the form Q(X | Y) = N(X_1, | f_1(Y)) * ... * N(X_n | f_n(Y)) -> mean field approx. of the inf comp proposal.
#This allows us to estimate mu(X_i | Y), sigma(X_i | Y). Then, we can normalize by doing (X_i - mu(X_i | Y = y)) / sigma(X_i, Y=y)
#This is useful because we don't want to have our neural net output large values. 
#The trivial thing to do, (X - X_bar)/sigma, is wrong because the normalization should differ depending on the observation (think about lin reg.)
 
class Normalizer():
    def __init__(self, g):
        super().__init__()
        self.graph = g
        self.ordered_vars = g.topological()
        self.ordered_latent_vars = [var for var in self.ordered_vars if var not in g.Graph["Y"].keys()]
        self.n = len(g.Graph["Y"]) #number of observations

        self.distributions = {var : Normal for var in self.ordered_latent_vars}
        self.constraints = {var : self.distributions[var].arg_constraints for var in self.ordered_latent_vars}
        
        self.links = {var : NeuralNetwork(dim_in = self.n, dim_out = self.distributions[var].NUM_PARAMS) for var in self.ordered_latent_vars}
        
        return
    
    def sample(self, y : tc.Tensor) -> dict:
        _sample = {}
        val = y #start by conditioning on y
        with tc.no_grad():
            for latent in self.ordered_latent_vars:
                #get output of NN. Should be of dimension == # dist params
                nn_out = self.links[latent].forward(val)
                #iterate over nn output and the associated support for each element (ex : mu, sigma for a Normal, hence sigma has (0,inf) support) and apply appropriate transform
                dist_params = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)] #apply constraint transform
                #sample from the conditional distribution
                s = self.distributions[latent](*dist_params).sample()
                _sample.update({latent : s})

        return _sample
    
    def log_prob(self, sample : tc.Tensor, y : tc.Tensor) -> tc.Tensor:
        """the sample should follow the topological order of the graphical model. (i.e. topological() method of custom graph class)"""
        lik = tc.tensor(0.0)
        val = y
        #sample comes in topological order
        for latent, x in zip(self.ordered_latent_vars, sample):
            nn_out = self.links[latent].forward(val)
            dist_params = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)]
            lik += self.distributions[latent](*dist_params).log_prob(x)

        return lik
    
    def get_params(self):
        params = []
        num_params = 0
        for latent, neuralnet in self.links.items():
            params += [*neuralnet.parameters()]
            num_params += sum(p.numel() for p in neuralnet.parameters() if p.requires_grad)
        self.num_params = num_params
        return params
    

    def normalize(self, x : tc.Tensor, y : tc.Tensor)-> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        """returns a normalized version of x | Y=y, so z | Y=y. x is in topological"""
        z = []
        mus = []
        sigmas = []
        with tc.no_grad():
            for i, latent in enumerate(self.ordered_latent_vars):
                nn_out = self.links[latent].forward(y)
                mu, sigma = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)] #apply constraint transform
                mus.append(mu)
                sigmas.append(sigma)
                z.append((x[i] - mu)/sigma)
        
        z = tc.stack(z)
        mus = tc.stack(mus)
        sigmas = tc.stack(sigmas)

        return z, mus, sigmas
    
    def denormalize(self, z : tc.Tensor, y : tc.Tensor)-> tc.Tensor:
        """returns a denormalized version of z, so x | Y=y. z is in topological"""
        x = []
        mus = []
        sigmas = []
        with tc.no_grad():
            for i, latent in enumerate(self.ordered_latent_vars):
                nn_out = self.links[latent].forward(y)
                mu, sigma = [transform_to(constr)(dist_param) for constr, dist_param in zip(self.distributions[latent].arg_constraints.values(), nn_out)] #apply constraint transform
                mus.append(mu)
                sigmas.append(sigma)
                x.append((z[i]*sigma + mu))
        
        x = tc.stack(x)
        mus = tc.stack(mus)
        sigmas = tc.stack(sigmas)
        jacobian = tc.prod(sigmas)
        return x
