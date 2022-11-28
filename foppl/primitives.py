import torch as tc
import ttvfast

def oneplanet(parameters, ith_transit):
    #if we have called oneplanet n%6 times, where 6 is then number of transit times, recompute the transit times (new set of parameters)
    if primitives["NUM_CALLS_ONEPLANET"]%6 == 0 or ith_transit==-1:
        mass, period, eccentricity, argument, meananomaly, inclination, longnode, stellar_mass = parameters
        
        planet = ttvfast.models.Planet(
            mass = mass,
            period = period,
            eccentricity = eccentricity,
            argument = argument,
            inclination = inclination,
            longnode = longnode,
            mean_anomaly = meananomaly
        )
        len_transit_time = 0
        total = 2500
        while len_transit_time<6:
            if total>=5000:
                raise Exception("Exceeded the total number of integration steps.")
            results = ttvfast.ttvfast([planet], stellar_mass, time=0.0, dt=0.5, total=total)
            idx, epochs, transit_times, rsky, vsky = results["positions"]
            transit_times = tc.tensor(transit_times).float()
            transit_times = transit_times[transit_times!=-2]
            len_transit_time = len(transit_times)
            total+=500

        if ith_transit==-1: #case where we just want to return the transit times for a set of parameters. 
            return transit_times[:6]
            
        primitives["CURRENT_TRANSIT_TIMES"] = transit_times

    primitives["NUM_CALLS_ONEPLANET"]+=1 #increment number of calls
    return primitives["CURRENT_TRANSIT_TIMES"][ith_transit.int().item()]


def _and(first, second):
    if first and second:
        return tc.tensor([True])
    else:
        return tc.tensor([False])

def _or(first, second):
    if first or second:
        return tc.tensor([True])
    else:
        return tc.tensor([False])


def equal(first, second):
    return tc.tensor(tc.equal(first, second))

def matrepmat(vecname, val , length):
    return tc.full((length.int(),), val)

def get(vec, pos):
    if tc.is_tensor(pos):
        return vec[int(pos.item())]
    else:
        return vec[int(pos)]

def put(vec, pos, val):
    if tc.is_tensor(int(pos)):
        vec[int(pos.item())] = val
    else:
        vec[int(pos)] = val
    return vec

def first(vec):
    return vec[0]

def second(vec):
    return vec[1]

def rest(vec):
    return vec[1:]

def last(vec):
    return vec[-1]

def append(vec, val):
    return tc.cat((vec, tc.tensor([val])))

def vector(*x):
    # NOTE: This must support both lists and vectors
    try:
        result = tc.stack(x)
    except:
        result = list(x)
    return result

def hashmap(*x):
    _keys = [key for key in x[0::2]]
    keys = []
    for key in _keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        keys.append(key)
    values = [value for value in x[1::2]]
    return dict(zip(keys, values))

class Dirac():
    def __init__(self, x0, atol=10e-1):
        self.x0 = x0
        self.atol = atol
        self.inf = 1e10

    def log_prob(self, x):
        if tc.isclose(x, self.x0, rtol=10e-5, atol=self.atol):
            return tc.tensor(0.0)
        else : 
            return tc.tensor(-self.inf).float()

    def sample(self):
        return self.x0


# Primative function dictionary
    

primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    'and': _and,
    'or': _or,

    # Math
    '+': tc.add,
    '-': tc.sub,
    '*': tc.mul,
    '/': tc.div,
    '=': equal,
    'sqrt': tc.sqrt,
    'abs': tc.abs,


    # Containers
    'vector': vector,
    'hash-map': hashmap,
    'get' : get,
    'put' : put,
    'first' : first,
    'second': second,
    'last' : last,
    'append' : append,
    'rest' : rest,
    'range': range,
    # ...

    # Matrices
    'mat-mul': tc.matmul,
    'mat-transpose': tc.t,
    'mat-tanh' : tc.tanh,
    'mat-add' : tc.add,
    'mat-repmat' : matrepmat,
    # ...

    # Distributions
    'normal': tc.distributions.Normal,
    'beta': tc.distributions.Beta,
    'exponential': tc.distributions.Exponential,
    'uniform-continuous': tc.distributions.Uniform,
    'discrete' : tc.distributions.Categorical,
    'dirichlet' : tc.distributions.Dirichlet,
    'gamma' : tc.distributions.Gamma,
    'flip' : tc.distributions.Bernoulli,
    'dirac': Dirac,

    #TTV functions and global variables
    'oneplanet' : oneplanet,
    'CURRENT_TRANSIT_TIMES' : None,
    'NUM_CALLS_ONEPLANET' : 0
    
}