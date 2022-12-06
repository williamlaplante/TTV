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
from normalizers import Normalizer
from model import ModelDataset



def inference_compilation(g : graph, program_name : str, num_samples = int(1e3), num_traces_training = int(1e6), num_workers=2, batch_size=int(5e3), num_epochs = 3, learning_rate=1e-4, train_normalizer = True, lstm=False, wandb_name = None, wandb_run = False):
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

    if train_normalizer:
        logdir = "./results/run_" + logdir_suffix + "_normalizer"
    else:
        logdir = "./results/run_" + logdir_suffix

    os.mkdir(logdir)

    if train_normalizer:
        num_samples_normalizer = int(1e5)
        batch_size_normalizer = 300

        print("\n=> Sampling from the model (normalizer)...")
        dataset = ModelDataset(g=g, num_samples=num_samples_normalizer)
        dataset.build_dataset()
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size_normalizer, shuffle=True, num_workers=num_workers)

        print("\n=> Training Normalizer...")
        losses_normalizers = []
        normalizer = Normalizer(g)
        normalizer_params = normalizer.get_params()
        print("\n=> Training {} parameters for the normalizer...".format(normalizer.num_params))

        normalizer_optimizer = tc.optim.Adam(normalizer_params)

        for i, data in enumerate(dataloader):
            normalizer_optimizer.zero_grad()
            
            X, y, lik = data

            #compute log probability
            try:
                log_normalizer = tc.stack([normalizer.log_prob(x, y) for (x,y) in zip(X, y)])
            except ValueError as e:
                print(e)
                continue

            loss_norm = -log_normalizer.mean()
            loss_norm.backward()
            normalizer_optimizer.step()

            if i%1 == 0 : print("=> Loss (Normalizer) : {}".format(loss_norm.detach().clone()))
            losses_normalizers.append(loss_norm.detach().clone())
        
        losses_normalizers = tc.stack(losses_normalizers)
        tc.save(losses_normalizers, logdir + "/losses_normalizer.pt")

        with open(logdir + "/normalizer.pkl", "wb") as f:
            pickle.dump(normalizer, f)


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
            try:
                log_Q = []
                for (x,y) in zip(X, y):
                    if train_normalizer: #if we have a normalizer, normalize x
                        z, mus, sigmas = normalizer.normalize(x, y) #x is in topological order -> z should be too
                    else:#else, z is just x
                        z = x
                    
                    log_Q.append(proposal.log_prob(z, y)) #Jacobian here? Don't think so

                log_Q = tc.stack(log_Q)
                    
            except ValueError as e:
                print(e)
                continue

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
    observed_dict = {} #a dict of observed values -> useful for computation of log prob
    for var in dataset.ordered_vars:
        if var in g.Graph["Y"].keys():
            observed.append(tc.tensor(g.Graph["Y"][var]).float())
            observed_dict.update({var : tc.tensor(g.Graph["Y"][var]).float()})

    observed = tc.stack(observed) #a tensor of observed values in topological order of the graph
    print("\n=> Observed values : {}".format(observed))

    #The final step is to perfrom sequential importance sampling by generating samples from the proposal
    print("\n=>Importance sampling with compilation artifact...\n")
    proposal_samples = []
    proposal_weights = []
    model_weights = []
    weights = []

    for i in range(num_samples):
        sample_dict = proposal.sample(y = observed) #sample of latent variables conditioned on Y=y
        sample_tensor = tc.stack([sample_dict[latent] for latent in dataset.latent_ordered_vars]) #tensor form of sample dict
        if normalizer : 
            sample_tensor = normalizer.denormalize(sample_tensor, observed) #normalize if normalizer is trained
            sample_dict = {latent : sample_tensor[i] for i, latent in enumerate(dataset.latent_ordered_vars)} #dict form of sample tensor

        proposal_samples.append(sample_tensor) #sample in the topological order
        model_weights.append(dataset.log_prob({**sample_dict, **observed_dict})) #compute log p(y|x)p(x)
        proposal_weights.append(proposal.log_prob(sample_tensor, observed)) #compute log q(x|y)
        weights.append(tc.exp(model_weights[i] - proposal_weights[i])) # w = exp (log p(x,y) - log q(x|y))

    proposal_samples = tc.stack(proposal_samples).detach().clone()
    proposal_weights = tc.stack(proposal_weights).detach().clone()
    model_weights = tc.stack(model_weights).detach().clone()
    weights = tc.stack(weights).detach().clone()
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