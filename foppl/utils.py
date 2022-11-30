import numpy as np
import torch as tc
import wandb

# Please add the functions below to your pre-existing utils.py file

def log_sample(sample, i, wandb_name):
    '''
    Log an individual sample to W&B
    '''
    if sample.dim() == 0:
        samples_dict = {wandb_name+'; epoch': i, wandb_name: sample}
    else:
        samples_dict = {wandb_name+'; epoch': i}
        for i, element in enumerate(sample):
            samples_dict[wandb_name+'; '+str(i)] = element
    wandb.log(samples_dict)


def wandb_plots(samples, program):
    '''
    Create W&B plots to upload for homework 2
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'slope', 'bias'])
        wandb_log['Program 2; slope'] = wandb.plot.histogram(table, value='slope', title='Program 2; slope')
        wandb_log['Program 2; bias'] = wandb.plot.histogram(table, value='bias', title='Program 2; bias')
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        data = np.array(samples)
        xs = np.linspace(0, data.shape[1]-1, num=data.shape[1])
        x = []; y = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x.append(xs[j])
                y.append(data[i, j])
        xedges = np.linspace(-0.5, data.shape[1]-0.5, data.shape[1]+1)
        yedges = np.linspace(-0.5, data.max()+0.5, int(data.max())+2)
        matrix, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[:-1]+0.5; ylabels = yedges[:-1]+0.5
        wandb_log['Program 3; heatmap'] = wandb.plots.HeatMap(xlabels, ylabels, matrix.T, show_text=True)
    elif program == 4:
        x_values = np.arange(samples.shape[1]+1)
        for y_values, name in zip([samples.mean(axis=0), samples.std(axis=0)], ['mean', 'std']):
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns=['position', name])
            title = 'Program 4; '+name
            wandb_log[title] = wandb.plot.line(table, 'position', name, title=title)
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)


def softplus(x, beta=1., threshold=20.):
    '''
    Original softplus: S(x) = (1/beta)*ln(1+exp(beta*x))
    '''
    s = tc.where(x<=threshold, tc.log(tc.exp(beta*x)+1.)/beta, x)
    return s


def inverse_softplus(s, beta=1., threshold=20.):
    '''
    Inverse softplus: x(S) = (1/beta)*ln(-1+exp(beta*S))
    '''
    x = tc.where(s<=threshold, tc.log(tc.exp(beta*s)-1.)/beta, s)
    return x


def log_params(variationals: list, i: int, program:int, wandb_name: str) -> None: 
    '''
    Log a set of variational-distribution parameters to W&B
    @params
        variationals: list of distributions corresponding to each sample node at each training step
        i: integer corresponding to epcoh
        wandb_name: string name of W&B run
    '''
    wandb_name_here = wandb_name+'program {} params'.format(program)
    samples_dict = {wandb_name_here+'; epoch': i}
    for node, distribution in variationals.items():
        params = [p.clone().detach().numpy() for p in distribution.params()]
        for i, param in enumerate(params):
            samples_dict[wandb_name_here+'; '+node+'; '+str(i)] = param
    wandb.log(samples_dict)


def log_loss(loss: dict, i: int, program, wandb_name: str) -> None:
    '''
    Log a set of losses corresponding to each node to W&B
    @params
        losses: list of losses corresponding to each sample node at each training step
        i: integer corresponding to epcoh
        wandb_name: string name of W&B run
    '''
    wandb_name_here = wandb_name+' loss; program ' + str(program)
    wandb_dict = {wandb_name_here+'; epoch': i}
    wandb_dict[wandb_name_here+';'] = loss
    wandb.log(wandb_dict)

def calculate_effective_sample_size(weights:tc.Tensor, verbose=False):
    '''
    Calculate the effective sample size via the importance weights
    '''
    N = len(weights)
    weights /= weights.sum()
    ESS = 1./(weights**2).sum()
    ESS = ESS.type(tc.float)
    if verbose:
        print('Sample size:', N)
        print('Effective sample size:', ESS)
        print('Fractional sample size:', ESS/N)
        print('Sum of weights:', weights.sum())
    return ESS
