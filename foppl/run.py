# Standard imports
import numpy as np
import torch as tc
from time import time
import wandb
import hydra


# Project imports
from daphne_utils import load_program
from tests import is_tol, run_probabilistic_test, load_truth
from general_sampling import get_sample, prior_samples
from evaluation_based_sampling import abstract_syntax_tree
from graph_based_sampling import graph
from utils import wandb_plots
from inference_compilation import inference_compilation
from importance_sampling import likelihood_weighting
from MH_gibbs import MH_gibbs
from HMC import HMC_sampling
from primitives import oneplanet

def create_class(ast_or_graph, mode):
    if mode == 'desugar':
        return abstract_syntax_tree(ast_or_graph)
    elif mode == 'graph':
        return graph(ast_or_graph)
    else:
        raise ValueError('Input type not recognised')

def run_tests(tests, mode, test_type, base_dir, daphne_dir, num_samples=int(1e4), max_p_value=1e-4, compile=False, verbose=False,):

    # File paths
    test_dir = base_dir+'/programs/tests/'+test_type+'/'
    daphne_test = lambda i: test_dir+'test_%d.daphne'%(i)
    json_test = lambda i: "./programs/tests/" + test_type + '/test_%d_%s.json'%(i, mode)
    truth_file = lambda i: './programs/tests/' + test_type + '/test_%d.truth'%(i)

    # Loop over tests
    print('Running '+test_type+' tests')
    for i in tests:
        print('Test %d starting'%i)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_test(i), json_test(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        truth = load_truth(truth_file(i))
        if verbose: print('Test truth:', truth)
        if test_type == 'deterministic':
            ret, _ = get_sample(ast_or_graph, mode, verbose=verbose)
            if verbose: print('Test result:', ret)
            try:
                assert(is_tol(ret, truth))
            except AssertionError:
                raise AssertionError('Return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast_or_graph))
        elif test_type == 'probabilistic':
            samples = []
            for _ in range(num_samples):
                sample, _ = get_sample(ast_or_graph, mode, verbose=verbose)
                samples.append(sample)
            p_val = run_probabilistic_test(samples, truth)
            print('p value:', p_val)
            assert(p_val > max_p_value)
        else:
            raise ValueError('Test type not recognised')
        print('Test %d passed'%i, '\n')
    print('All '+test_type+' tests passed\n')

def run_programs(programs, inference_methods, mode, prog_set, base_dir, daphne_dir, num_traces_training, batch_size=int(256), num_samples=int(1e3), learning_rate=1e-4, num_epochs=2, lstm=True, tmax=None, compile=False, wandb_run=False, verbose=False):

    # File paths
    prog_dir = base_dir+'/programs/'+prog_set+'/'
    daphne_prog = lambda i: prog_dir + '%d.daphne'%(i)
    json_prog = lambda i: prog_dir + '%d_%s.json'%(i, mode)

    for i in programs:
        print("\n=> Running program {}...".format(i))

        #load the program
        ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)

        for inference_method in inference_methods:
            
            print("\n=> Running {} inference engine...".format(inference_method))
            wandb_name = 'Program {} samples; {} inference'.format(i, inference_method) if wandb_run else None

            if inference_method == "importance_sampling":
                t_start = time()
                samples = likelihood_weighting(g=ast_or_graph, num_samples=num_samples, wandb_name=wandb_name, verbose=verbose)
                samples = tc.stack(samples)
                print("mean : {}, standard deviation : {}".format(samples.mean(axis=0), samples.std(axis=0)))

                if i==6 : 
                    print("Transit times : ", oneplanet(tc.cat([samples.mean(axis=0), tc.tensor([90.0, 0.0, 0.95])]), -1))
                    print("True Transit times : ", [66.1638, 274.0242, 482.0364, 689.8718, 897.8566, 1105.8455])

                t_finish = time()
                print('Time taken [s]:', t_finish-t_start)

            if inference_method == "MH_gibbs":
                t_start = time()
                samples = MH_gibbs(g=ast_or_graph, num_samples=num_samples, wandb_name=wandb_name, verbose=verbose)
                samples = tc.stack(samples)
                samples = samples[int(num_samples/5):]
                print("mean : {}, standard deviation : {}".format(samples.mean(axis=0), samples.std(axis=0)))

                if i==6 : 
                    print("Transit times : ", oneplanet(tc.cat([samples.mean(axis=0), tc.tensor([90.0, 0.0, 0.95])]), -1))
                    print("True Transit times : ", [66.1638, 274.0242, 482.0364, 689.8718, 897.8566, 1105.8455])

                t_finish = time()
                print('Time taken [s]:', t_finish-t_start)
            
            if inference_method == "HMC":
                t_start = time()
                samples = HMC_sampling(g=ast_or_graph, num_samples=num_samples, wandb_name=wandb_name, verbose=verbose)
                samples = tc.stack(samples).detach()
                print("mean : {}, standard deviation : {}".format(samples.mean(axis=0), samples.std(axis=0)))
                if i==6 : 
                    print("Transit times : ", oneplanet(tc.cat([samples.mean(axis=0), tc.tensor([90.0, 0.0, 0.95])]), -1))
                    print("True Transit times : ", [66.1638, 274.0242, 482.0364, 689.8718, 897.8566, 1105.8455])

                t_finish = time()
                print('Time taken [s]:', t_finish-t_start)


            if inference_method == "inference_compilation":
                samples = inference_compilation(g = ast_or_graph, num_samples=num_samples, num_traces_training=num_traces_training, learning_rate=learning_rate, batch_size=batch_size, program_name="program " + str(i), num_epochs=num_epochs, lstm=lstm, wandb_name=wandb_name, wandb_run=wandb_run)
                samples = tc.stack(samples)
                print("mean : {}, standard deviation : {}".format(samples.mean(axis=0), samples.std(axis=0)))
                t_finish = time()

        print('Finished program {}\n'.format(i))


@hydra.main(version_base=None, config_path='', config_name='config')
def run_all(cfg):

    # Configuration
    wandb_run = cfg['wandb_run']
    mode = cfg['mode']
    num_samples = int(cfg['num_samples'])
    compile = cfg['compile']
    base_dir = cfg['base_dir']
    daphne_dir = cfg['daphne_dir']
    prog_set = cfg["prog_set"]
    entity = cfg["entity"]
    save = cfg["save"]
    verbose = cfg["verbose"]
    inference_methods = cfg["inference_methods"]
    num_traces_training = int(cfg["num_traces_training"])
    batch_size = int(cfg["batch_size"])
    num_epochs = int(cfg["num_epochs"])
    learning_rate = cfg["learning_rate"]
    lstm = cfg["lstm"]

    # Initialize W&B
    if wandb_run: wandb.init(project=prog_set +'-'+mode, entity=entity)
    
    # Deterministic tests
    tests = cfg['deterministic_tests']
    run_tests(tests, mode=mode, test_type='deterministic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile)

    # Proababilistic tests
    tests = cfg['probabilistic_tests']
    run_tests(tests, mode=mode, test_type='probabilistic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile)

    # Programs
    programs = cfg[prog_set + '_programs']

    run_programs(programs, inference_methods, mode=mode, prog_set=prog_set, base_dir=base_dir, daphne_dir=daphne_dir, num_samples=num_samples, num_traces_training=num_traces_training, learning_rate=learning_rate, lstm=lstm, batch_size=batch_size, num_epochs=num_epochs, compile=compile, wandb_run=wandb_run, verbose=verbose)

    # Finalize W&B
    if wandb_run: wandb.finish()

if __name__ == '__main__':
    run_all()