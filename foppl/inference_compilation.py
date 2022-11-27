import itertools
import json

import torch as tc
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from proposal import Proposal
from graph_based_sampling import Eval, graph, standard_env
from utils import log_loss, log_params, log_sample


class ModelDataset(Dataset):

    def __init__(self, g, num_samples = int(1e4)):
        self.graph = g
        self.ordered_vars = g.topological()
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



def inference_compilation(g, program_name, num_samples = int(1e3), num_traces_training = int(1e6), num_workers=2, batch_size=int(5e3), num_epochs = 3, learning_rate=1e-4, lstm=True, wandb_name = None, wandb_run = False):
    """
    Parameters
    ----------
    g : graphical model
    program : name of the graphical model
    num_samples : Total number of samples the proposal distribution will see
    batch_size : number of samples per training iterations
    num_workers : No idea

    """
    if not isinstance(g, graph):
        raise Exception("Inference compilation has not yet been implemented for non-graph programs.")
    
    print("=> Sampling from the model...")
    dataset = ModelDataset(g=g, num_samples=num_traces_training)
    dataset.build_dataset()

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    proposal = Proposal(g, lstm=lstm)
    params = proposal.get_params()
    optimizer = tc.optim.Adam(params, lr=learning_rate)

    print("\n=> Training {} parameters | LSTM : {}".format(proposal.num_params, lstm))
    print("=> Running {} epochs...".format(num_epochs+1))
    print("=> Learning Rate : {} | Batch Size : {}".format(learning_rate, batch_size))
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("\n=> Epoch {}...\n".format(epoch))

        for i, data in enumerate(dataloader):
            #zero parameter gradients
            optimizer.zero_grad()

            # get the data
            X, y, lik = data

            if i==0 and epoch==0:
                print("=> Single sample : X = {} | Y = {}\n".format(X[0], y[0]))
 
            #compute log probability of proposal with 
            log_Q = tc.stack([proposal.log_prob(x, y) for (x,y) in zip(X, y)])

            loss = -log_Q.mean()
            loss.backward()
            optimizer.step()

            if i%1 == 0 : print("=> Loss : {}".format(loss.detach().clone()))
            if wandb_run : log_loss(loss.detach().clone(), i, program=program_name, wandb_name=wandb_name)

    print('Finished Training\n')

    observed = []
    for var in dataset.ordered_vars:
        if var in g.Graph["Y"].keys():
            observed.append(tc.tensor(g.Graph["Y"][var]).float())

    observed = tc.stack(observed)

    print("\n=> Observed values : {}".format(observed))
    samples = [tc.stack([val for val in proposal.sample(y = observed).values()]) for _ in range(num_samples)]
    return samples