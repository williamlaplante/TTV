import itertools
import json
import os
from datetime import datetime
import pickle
import torch as tc
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from proposal import Proposal
from graph_based_sampling import Eval, graph, standard_env
from utils import log_loss, log_params, log_sample, calculate_effective_sample_size


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
        sigma = {"logW":tc.tensor(0.0), "logP":tc.tensor(0.0), "logJoint":tc.tensor(0.0)}
        env = standard_env()
        env.update(sample)
        lik = tc.tensor(0.0)
        for var in self.latent_ordered_vars:
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



def inference_compilation(g : graph, program_name : str, num_samples = int(1e3), num_traces_training = int(1e6), num_workers=2, batch_size=int(5e3), num_epochs = 3, learning_rate=1e-4, lstm=False, wandb_name = None, wandb_run = False):
    """
    Parameters
    ----------
    g : graphical model
    program : name of the graphical model
    num_samples : Total number of samples the proposal distribution will see
    batch_size : number of samples per training iterations
    num_workers : No idea

    """
    #if its not a graph class, we don't want it
    if not isinstance(g, graph):
        raise Exception("Inference compilation has not yet been implemented for non-graph programs.")
    
    #Initialize the folder in which we will save information for future diagnostics
    logdir_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = "./results/run_" + logdir_suffix
    os.mkdir(logdir)

    #Get samples from the model. We want to fit these samples
    print("=> Sampling from the model...")
    dataset = ModelDataset(g=g, num_samples=num_traces_training)
    dataset.build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #Initialize the proposal distribution
    proposal = Proposal(g, lstm=lstm)
    params = proposal.get_params()
    optimizer = tc.optim.Adam(params, lr=learning_rate)
    losses = []

    #Printing some interesting information for diagnostic purposes
    print("\n=> Training {} parameters | LSTM : {}".format(proposal.num_params, lstm))
    print("=> Running {} epochs...".format(num_epochs+1))
    print("=> Learning Rate : {} | Batch Size : {}".format(learning_rate, batch_size))

    #Train the weights charaterizing the proposal distributions
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("\n=> Epoch {}...\n".format(epoch))
        losses_epoch = []

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

            #log the loss
            if i%1 == 0 : print("=> Loss : {}".format(loss.detach().clone()))
            losses_epoch.append(loss.detach().clone())
            if wandb_run : log_loss(loss.detach().clone(), i, program=program_name, wandb_name=wandb_name)

        #save the proposal at that epoch for future diagnostics
        with open(logdir + "/proposal_epoch{}.pkl".format(epoch), "wb") as f:
            pickle.dump(proposal, f)

        losses.append(losses_epoch)
        print("\n=> Expected loss for epoch {} : {}".format(epoch, tc.stack(losses_epoch).mean()))

    print('Finished Training\n')

    #At this point we have P(X | Y). We now want to collect Y=y to make inference on P(X | Y=y)
    observed = []
    for var in dataset.ordered_vars:
        if var in g.Graph["Y"].keys():
            observed.append(tc.tensor(g.Graph["Y"][var]).float())
    observed = tc.stack(observed)
    print("\n=> Observed values : {}".format(observed))

    #The final step is to perfrom sequential importance sampling by generating samples from the proposal
    print("\n=>Importance sampling with compilation artifact...\n")
    proposal_samples = []
    proposal_weights = []
    model_weights = []
    weights = []

    for i in range(num_samples):
        sample_dict = proposal.sample(y = observed)
        sample_tensor = tc.stack([sample_dict[latent] for latent in dataset.latent_ordered_vars]) #topological order
        proposal_samples.append(sample_tensor) #sample in the topological order
        model_weights.append(dataset.log_prob(sample_dict))
        proposal_weights.append(proposal.log_prob(sample_tensor, observed))
        weights.append(tc.exp(model_weights[i] - proposal_weights[i]))

    proposal_samples = tc.stack(proposal_samples)
    proposal_weights = tc.stack(proposal_weights)
    model_weights = tc.stack(model_weights)
    weights = tc.stack(weights)
    calculate_effective_sample_size(weights, verbose=True)


    #saving everything in a folder
    tc.save(proposal_samples, logdir + "/proposal_samples.pt")
    tc.save(proposal_weights, logdir + "/proposal_weights.pt")
    tc.save(model_weights, logdir + "/model_weights.pt")
    tc.save(weights, logdir + "/weights.pt")
    tc.save(losses, logdir + "/losses.pt")
    
    with open(logdir + "/graphical_model.pkl", "wb") as f:
        pickle.dump(g, f)
    
    with open(logdir + "/model.pkl", "wb") as f:
        pickle.dump(ModelDataset(g=g, num_samples=num_traces_training), f)

    return proposal_samples