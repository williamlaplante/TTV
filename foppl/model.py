import torch as tc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from graph_based_sampling import Eval, graph, standard_env


class ModelDataset(Dataset):

    def __init__(self, g, num_samples = int(1e4)):
        self.graph = g
        self.ordered_vars = g.topological()
        self.latent_ordered_vars = [var for var in self.ordered_vars if var not in g.Graph["Y"].keys()]
        self.num_samples = num_samples
        self.X = []
        self.Y = []
        self.lik = []
        return
    
    def get_sample(self, dict_form = False):
        sigma = {"logW":tc.tensor(0.0), "logP":tc.tensor(0.0), "logJoint":tc.tensor(0.0)}
        env = standard_env()
        sample = {}
        lik = tc.tensor(0.0)
        for var in self.ordered_vars:
            dist, _ = Eval(self.graph.Graph["P"][var][1], sigma, env)
            s = {var : dist.sample()}
            lik += dist.log_prob(s[var])
            env.update(s)
            sample.update(s)
        
        if dict_form:
            return sample, lik

        else:
            x = []
            y = []
            for key, val in sample.items(): #note : this follows topological order
                if key.startswith("sample"): x.append(val)
                elif key.startswith("observe") : y.append(val)

            x = tc.stack(x) #in topological order
            y = tc.stack(y) #in topological order

            return x, y, lik
    
    def log_prob(self, sample : dict) -> tc.Tensor:
        """Computes log p(y|x)p(x) using a sample in dict form | example : {"sample3" : 4.0, ...}"""
        sigma = {"logW":tc.tensor(0.0), "logP":tc.tensor(0.0), "logJoint":tc.tensor(0.0)}
        env = standard_env()
        env.update(sample)
        lik = tc.tensor(0.0)
        for var in self.ordered_vars:
            dist, _ = Eval(self.graph.Graph["P"][var][1], sigma, env)
            lik+=dist.log_prob(sample[var])
    
        return lik

    def build_dataset(self):
        self.X = []
        self.Y = []
        self.lik = []
        for i in tqdm(range(self.num_samples)):
            x, y, lik = self.get_sample()
            self.X.append(x)
            self.Y.append(y)
            self.lik.append(lik)
        self.X = tc.stack(self.X) #cols are in topological order
        self.Y = tc.stack(self.Y) #cols are in topological order
        self.lik = tc.stack(self.lik)
        return

    def __getitem__(self, idx):
        #dataset[idx] -> we don't care about the idx; 
        return self.X[idx], self.Y[idx], self.lik[idx]

    def __len__(self):
        # len(dataset)
        return self.num_samples

