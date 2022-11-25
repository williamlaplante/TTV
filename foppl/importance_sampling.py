import json
import torch as tc
from tqdm import tqdm

from graph_based_sampling import graph
from general_sampling import get_sample
from utils import log_sample


def likelihood_weighting(g : graph, num_samples=int(1e3), wandb_name=None, verbose=False):

        if not isinstance(g, graph):
            raise Exception("Not yet implemented non-graph programs.")

        samples = []
        log_weights = []
        for i in tqdm(range(num_samples)):
            ret, sig = get_sample(g, mode='graph', verbose=verbose)
            samples.append(ret)
            log_weights.append(sig["logW"])

        log_weights = tc.stack(log_weights).type(tc.float)
        samples = tc.stack(samples).type(tc.float)

        weights = tc.exp(log_weights)
        normalized_weights = tc.div(weights, weights.sum())

        if tc.isclose(weights.sum(), tc.tensor([0.0])):
            raise Exception("Likelihood weights are all 0's.")

        idx = tc.distributions.Categorical(normalized_weights).sample(tc.Size([num_samples]))

        resample = []
        for i in idx:
            s = samples[i.item()]
            resample.append(s)
            if wandb_name is not None: log_sample(s, i, wandb_name=wandb_name)

        return resample