import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import src.model
import numpy as np
import time
from src.client import Client
from src.server import Server
import sys
from numpy import savetxt
import os
import json
import src.dataloaders
from src.algorithms import *
import logging
from losses_replot import plot_losses
from src.dataloaders import *
import string
import random
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

import pickle
import matplotlib
import cProfile
import pstats

plt.rcParams['font.family'] = 'sans-serif' 
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append('ByzLibrary')

from robust_aggregators import RobustAggregator
from byz_attacks import ByzantineAttack

import argparse

parser = argparse.ArgumentParser(description='Byzantine robustness with personalized learning')
parser.add_argument('--n', type=int, default=20, metavar='N',
                    help='number of workers (default: 20)')
parser.add_argument('--m', type=int, default=64, metavar='M',
                    help='number of datapoints per worker (default: 64)')
parser.add_argument('--test_m', type=int, default=2000, metavar='M',
				   	help='number of datapoints for the test set (default: 2000)')
parser.add_argument('--f', type=int, default=0, metavar='F',
                    help='number of faulty workers (default: 0)')
parser.add_argument('--T', type=int, default=100, metavar='T',
                    help='number of communication rounds (default: 100)')
parser.add_argument('--heterogeneity', type=str, default="homogeneous", metavar='H',
                    help='type of heterogeneity (default: homogeneous)')
parser.add_argument('--alpha', type=float, default=1, metavar='ALPHA',
                    help='alpha parameter for heterogeneity (default: 1)')
parser.add_argument('--attack', type=str, default="SF", metavar='AT',
                    help='type of attack (default: SF)')
parser.add_argument('--agg', type=str, default="trmean", metavar='AG',
                    help='type of aggregation (default: trmean)')
parser.add_argument('--nb_run', type=int, default=1, metavar='N',
                    help='number of runs (default: 1)')
parser.add_argument('--dataset', type=str, default="mnist", metavar='D',
				   	help='dataset used (default: mnist)')
parser.add_argument('--model', type=str, default="cnn", metavar='M',
                    help='type of model (default: cnn)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
					help='learning rate (default: 0.05)')
parser.add_argument('--batch_size', type=int, default=64, metavar='B',
					help='batch size (default: 64)')
parser.add_argument('--nb_main_client', type=int, default=1, metavar='NMC',
					help='number of main clients (default: 1)')
parser.add_argument('--nb_classes', type=int, default=2, metavar='NC',)
parser.add_argument('--algo', type=str, default="IPGD", metavar='A')
parser.add_argument('--eval_every', type=int, default=10, metavar='EE')
args = parser.parse_args()

print(args)


#torch.manual_seed(random_seed)
random_seed = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


config = {
	'n': args.n,
	'm': args.m,
	'test_m': args.test_m,
	'f': args.f,
	'T': args.T,
	'heterogeneity': args.heterogeneity,
	'alpha': args.alpha,
	'attack': args.attack,
	'agg': args.agg,
	'nb_run': args.nb_run,
	'dataset': args.dataset,
	'model': args.model,
	'lr' : args.lr,
	'gamma' : 0.2,
	'nb_classes' : args.nb_classes,  # TODO change to 10 for mnist
	'optim' : 'gd',
	'device' : device, 
	'seed' : random_seed,
	'batch_size': args.batch_size,
	'nb_main_client': args.nb_main_client,
	'eval_every':args.eval_every,
	'algo' : args.algo
}
print(config)

nb_honest = config['n'] - config['f']

# Verifying if folder exists 
dataset_folder = f'./experiments/{config["dataset"]}'
if not os.path.isdir(dataset_folder):
	os.mkdir(dataset_folder)
heterogeneity_folder = f'./experiments/{config["dataset"]}/{config["heterogeneity"]}'
if not os.path.isdir(heterogeneity_folder):
	os.mkdir(heterogeneity_folder)




# Settin up the folder where to store the plots and the experiments
save_folder = f'./experiments/{config["dataset"]}/{config["heterogeneity"]}/{config["attack"]}/n_{config["n"]}_m_{config["m"]}_f_{config["f"]}_T_{config["T"]}_runs_{config["nb_run"]}_alpha_{config["alpha"]}'


if os.path.isdir(save_folder):
	save_folder = save_folder + '_' + random.choice(string.ascii_letters)

if os.path.isdir(save_folder):
	save_folder = save_folder + '_' + random.choice(string.ascii_letters)

if os.path.isdir(save_folder):
	raise ValueError("Folder already exists, try again")

os.makedirs(save_folder)

logging.basicConfig(filename=os.path.join(save_folder, "log.log"), level=logging.INFO)
print("Created new directory ", save_folder)
logging.info(f"Created new directory {save_folder}")

with open(save_folder+'/config.json', 'w') as fp:
	tmp = device
	config['device'] = str(tmp)
	json.dump(config, fp, indent=4)
	config['device'] = tmp

#server = Server(config)

dataloading_func = heterogneity[config["heterogeneity"]]

dataloaders = dataloading_func(config,3,6)



print('Initial dataloading done')
logging.info("Initial dataloading done")
main_client = Client(config, 0, dataloaders[0])
model_size = main_client.model_size

results = [] # Contains average test accuracy for each lambda 
f1_scores= []
losses = []
accuracy_chechpoints = []
f1_checkpoints = []
runtimes = []


np.random.seed(random_seed)
seeds = np.random.randint(0, 100000, config['nb_run'])

#profiler = cProfile.Profile()
#profiler.enable()

for run in range(config['nb_run']):
	start = time.time()
	print("Run ", run)
	dataloaders =  dataloading_func(config, config['m'], config["test_m"], batch_size = config["batch_size"], alpha = config["alpha"], seed = seeds[run], save_folder = save_folder) 
	#ipgd = IPGD(config, dataloaders, model_size)
	algo = algorithms_dict[config['algo']](config, dataloaders, model_size, return_loss = True)#, lams = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
	#results_array, f1_score,losses_array = algo.run()
	results_array,f1_score, losses_array, results_array_checkpoints, f1_score_checkpoints= algo.run()
	results.append(results_array)
	losses.append(losses_array)
	f1_scores.append(f1_score)
	accuracy_chechpoints.append(results_array_checkpoints)
	f1_checkpoints.append(f1_score_checkpoints)
	end = time.time()
	runtimes.append(end-start)
	#print("Results average", results_array.mean())

results = np.array(results)
f1_scores = np.array(f1_scores)
losses = np.array(losses)
accuracy_chechpoints = np.array(accuracy_chechpoints)
f1_checkpoints = np.array(f1_checkpoints)
#profiler.disable()
#stats = pstats.Stats(profiler).sort_stats('cumulative')
#stats.dump_stats(save_folder+'/profile.prof')
#stats.print_stats()


print('runtime', runtimes)
logging.info(f"runtime {runtimes}")

print("results", results)
logging.info(f'results {results}')


print("Average accuracies for last run " ,results_array.mean(axis=1)) # THe last results_array
print("f1 score for last run ",f1_score)

print("Average accuracies ", np.mean(results, axis = (0,2)))
logging.info(f"Average accuracies {np.mean(results, axis = (0,2))}")


print("F1 score", f1_scores.mean(axis=(0,2)))
logging.info(f"f1 score {f1_scores.mean(axis=(0,2))}")


# Saving the results_array in a pickle file # TODO : change to results instead of results_array
with open(save_folder+'/results_array.pickle', 'wb') as handle:
	pickle.dump(results_array, handle)

with open(save_folder+'/results.pickle', 'wb') as handle:
	pickle.dump(results, handle)

with open(save_folder+'/losses_array.pickle', 'wb') as handle:
	pickle.dump(losses_array, handle)

with open(save_folder+'/f1_scores.pickle', 'wb') as handle:
	pickle.dump(f1_scores, handle)

with open(save_folder+'/accuracy_checkpoints.pickle', 'wb') as handle:
	pickle.dump(accuracy_chechpoints, handle)

with open(save_folder+'/f1_checkpoints.pickle', 'wb') as handle:
	pickle.dump(f1_checkpoints, handle)
	

# Plotting the losses

plot_losses(save_folder)


# Plotting error bars plot for the test accuracy 

"""plt.errorbar([0,0.2,0.4,0.6,0.8,1], np.mean(results_array, axis = 0), yerr = np.std(results_array, axis = 0), fmt='o', ecolor='orangered', capsize=3 )
plt.xlabel(r"$\lambda$")
plt.ylabel("Test accuracy on local dataset")
plt.title(rf"Test accuracy for first client, $n = {config['n']}$, $f = {config['f']}$ ")
plt.ylim(65, 100)
plt.savefig(save_folder+'/plot.png')
plt.show()"""


# Plotting the test accuracy for the first client
"""plt.plot(config["lams"], results_list)
plt.xlabel(r"$\lambda$")
plt.ylabel("Test accuracy on local dataset")
plt.title(rf"Test accuracy for first client, $n = {config['n']}$, $f = {config['f']}$ ")
plt.savefig(save_folder+'/plot_1.png')
plt.show()
"""
