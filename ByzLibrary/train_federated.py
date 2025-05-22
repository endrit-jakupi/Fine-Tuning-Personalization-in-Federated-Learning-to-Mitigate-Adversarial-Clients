# coding: utf-8
###
 # @file train_federated.py
 # @author John Stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
###

import tools, dataset, misc
tools.success("Module loading...")
import torch, argparse, sys, signal, pathlib, random
from worker_federated import Worker_Fed
from server_federated import Server_Fed

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandline():
	""" Parse the command-line and perform checks.
	Returns:
		Parsed configuration
	"""
	# Description
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--seed",
		type=int,
		default=-1,
		help="Fixed seed to use for reproducibility purpose, negative for random seed")
	parser.add_argument("--device",
		type=str,
		default="auto",
		help="Device on which to run the experiment, \"auto\" by default")
	parser.add_argument("--nb-steps",
		type=int,
		default=1000,
		help="Number of (additional) training steps to do, negative for no limit")
	parser.add_argument("--nb-workers",
		type=int,
		default=15,
		help="Total number of worker machines")
	parser.add_argument("--nb-decl-byz",
		type=int,
		default=0,
		help="Number of Byzantine worker(s) to support")
	parser.add_argument("--nb-real-byz",
		type=int,
		default=0,
		help="Number of actual Byzantine worker(s)")
	parser.add_argument("--gar",
		type=str,
		default="average",
		help="(Byzantine-resilient) aggregation rule to use")
	parser.add_argument("--gar-second",
		type=str,
		default=None,
		help="Second (Byzantine-resilient) aggregation rule to use on top of bucketing or NNM")
	parser.add_argument("--bucket-size",
		type=int,
		default=1,
		help="Size of buckets (i.e., number of gradients to average per bucket) in case of bucketing technique")
	parser.add_argument("--attack",
		type=str,
		default=None,
		help="Attack to use")
	parser.add_argument("--model",
		type=str,
		default="cnn_mnist",
		help="Model to train")
	parser.add_argument("--loss",
		type=str,
		default="NLLLoss",
		help="Loss to use")
	parser.add_argument("--dataset",
		type=str,
		default="mnist",
		help="Dataset to use")
	parser.add_argument("--batch-size",
		type=int,
		default=25,
		help="Batch-size to use for training")
	parser.add_argument("--batch-size-test",
		type=int,
		default=100,
		help="Batch-size to use for testing")
	parser.add_argument("--local-steps",
		type=int,
		default=1,
		help="Number of local steps to be performed at every worker")
	parser.add_argument("--learning-rate",
		type=float,
		default=0.5,
		help="Learning rate to use for training")
	parser.add_argument("--learning-rate-decay",
		type=int,
		default=5000,
		help="Learning rate hyperbolic half-decay time, non-positive for no decay")
	parser.add_argument("--learning-rate-decay-delta",
		type=int,
		default=1,
		help="How many steps between two learning rate updates, must be a positive integer")
	parser.add_argument("--momentum-worker",
		type=float,
		default=0,
		help="Momentum on workers to use for training")
	parser.add_argument("--weight-decay",
		type=float,
		default=0,
		help="Weight decay (L2-regularization) to use for training")
	#JS: L2 gradient clipping at worker
	parser.add_argument("--gradient-clip",
		type=float,
		default=None,
		help="Maximum L2-norm above which clipping occurs for the estimated gradients")
	parser.add_argument("--result-directory",
		type=str,
		default=None,
		help="Path of the directory in which to save the experiment results (loss, cross-accuracy, ...) and checkpoints, empty for no saving")
	parser.add_argument("--evaluation-delta",
		type=int,
		default=50,
		help="How many training steps between model evaluations, 0 for no evaluation")
    #JS: argument for heterogeneous setting
	parser.add_argument("--hetero",
		action="store_true",
		default=False,
		help="Heterogeneous setting")
    #JS: argument for number of labels of dataset (useful for heterogeneity + labelflipping)
	parser.add_argument("--numb-labels",
		type=int,
		default=None,
		help="Number of labels of dataset")
    #JS: argument for distinct datasets for honest workers
	parser.add_argument("--distinct-data",
		action="store_true",
		default=False,
		help="Distinct datasets for honest workers (e.g., privacy setting)")
    #JS: argument for sampling honest data using Dirichlet distribution
	parser.add_argument("--dirichlet-alpha",
		type=float,
		default=None,
		help="The alpha parameter for distribution the data among honest workers using Dirichlet")
    #JS: argument for number of datapoints per honest worker, in case of distinct datasets
	parser.add_argument("--nb-datapoints",
		type=int,
		default=None,
		help="Number of datapoints per honest worker in case of distinct datasets setting (e.g., privacy setting)")
	#JS: argument for precision of the quantization (before encryption), i.e., number of bits
	parser.add_argument("--bit-precision",
		type=int,
		default=None,
		help="Number of bits (precision) of quantization, must be greater than 1")
    #JS: argument for clamping gradients prior to quantization
	parser.add_argument("--parameter-clamp",
		type=float,
		default=None,
		help="Maximum coordinate value above which the parameters are clamped (for quantization)")
    #JS: argument for gradient subsampling on the server
	parser.add_argument("--subsampling-ratio",
		type=float,
		default=1,
		help="Ratio of workers to be subsampled in every iteration")
	# Parse command line
	return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
	args = process_commandline()
	# Count the number of real honest workers
	args.nb_honests = args.nb_workers - args.nb_real_byz

	cmdline_config = "Configuration" + misc.print_conf((
		("Reproducibility", "not enforced" if args.seed < 0 else (f"enforced (seed {args.seed})")),
		("#workers", args.nb_workers),
		("#declared Byz.", args.nb_decl_byz),
		("#actually Byz.", args.nb_real_byz),
		("Model", args.model),
		("Dataset", (
			("Name", args.dataset),
			("Batch size", (
				("Training", args.batch_size),
				("Testing", args.batch_size_test))))),
		("Loss", (
			("Name", args.loss),
			("L2-regularization", "none" if args.weight_decay is None else f"{args.weight_decay}"))),
		("Optimizer", (
			("Name", "sgd"),
			("Learning rate", args.learning_rate),
			("Momentum", args.momentum_worker))),
		("Gradient clip", "no" if args.gradient_clip is None else f"{args.gradient_clip}"),
		("Attack", args.attack),
		("Aggregation", args.gar),
		("Second Aggregation", args.gar_second),
        ("Extreme Heterogeneity", "yes" if args.hetero else "no"),
        ("Distinct datasets for honest workers", "yes" if args.distinct_data else "no"),
        ("Dirichlet distribution", "alpha = " + str(args.dirichlet_alpha) if args.dirichlet_alpha is not None else "no"),
		("Subsampling Ratio",  args.subsampling_ratio),
		("Encryption", "yes" if args.bit_precision is not None else "no")))
	print(cmdline_config)

# ---------------------------------------------------------------------------- #
# Setup
tools.success("Experiment setup...")

with tools.Context("setup", "info"):
	# Enforce reproducibility if asked (see https://pytorch.org/docs/stable/notes/randomness.html)
	reproducible = args.seed >= 0
	if reproducible:
		torch.manual_seed(args.seed)
		random.seed(args.seed)
		import numpy
		numpy.random.seed(args.seed)
	torch.backends.cudnn.deterministic = reproducible
	torch.backends.cudnn.benchmark   = not reproducible

	# JS: Create train (one for every honest worker) and test data loaders
	train_loader_dict, test_loader = dataset.make_train_test_datasets(args.dataset, heterogeneity=args.hetero,
								  numb_labels=args.numb_labels, alpha_dirichlet=args.dirichlet_alpha, distinct_datasets=args.distinct_data,
								  nb_datapoints=args.nb_datapoints, honest_workers=args.nb_honests, train_batch=args.batch_size, test_batch=args.batch_size_test)

	# Make the result directory (if requested)
	if args.result_directory is not None:
		resdir = pathlib.Path(args.result_directory).resolve()
		resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
		args.result_directory = resdir

# ---------------------------------------------------------------------------- #
# Training
tools.success("Training...")

# Training until limit or stopped
with tools.Context("training", "info"):
	#JS: Initialize the file in which to store the accuracies, if requested
	fd_eval = (args.result_directory / "eval").open("w") if args.result_directory is not None else None
	if fd_eval is not None:
		misc.make_result_file(fd_eval, ["Step number", "Cross-accuracy"])

	#JS: Initialize workers
	Workers = list()
	for worker_id in range(args.nb_honests):
		#JS: Instantiate worker i
		worker_i = Worker_Fed(train_loader_dict[worker_id], test_loader if worker_id == 0 else None, args.local_steps, args.model,
						args.learning_rate, args.weight_decay, args.loss, args.momentum_worker, args.gradient_clip, args.bit_precision,
						args.parameter_clamp, args.device)
		#JS: make the workers agree on the initial model
		if worker_id > 0:
			worker_i.model.load_state_dict(Workers[0].model.state_dict())
		Workers.append(worker_i)

	#JS: Initialize the server
	server = Server_Fed(Workers[0].model_size, args.gar, args.gar_second, args.learning_rate, args.learning_rate_decay,
					 args.learning_rate_decay_delta, args.device, args.nb_workers, args.nb_decl_byz, args.bucket_size,
					 args.subsampling_ratio, args.bit_precision, args.parameter_clamp)

	#JS: Start the training
	current_step = 0
	while not exit_is_requested() and current_step <= args.nb_steps:
		#JS: Evaluate the model if milestone is reached
		milestone_evaluation = args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0
		if milestone_evaluation:
			accuracy = Workers[0].compute_accuracy()
			print(f"Accuracy (step {current_step})... {accuracy * 100.:.2f}%.")
			# Store the evaluation result
			if fd_eval is not None:
				misc.store_result(fd_eval, current_step, accuracy)

		# ------------------------------------------------------------------------ #
		#JS: subsample workers at server and do local steps
		subsampled_workers = server.subsample_workers()
		subsampled_parameters = [Workers[worker_id].update_parameters() for worker_id in subsampled_workers]

		# Aggregate the subsampled parameters at the server
		aggregate_params = server.aggregate_parameters(subsampled_parameters)

		#JS: compute new learning rate at server
		new_learning_rate = server.update_learning_rate(current_step)
		#JS: broadcast the model parameters to all workers (and also send new learning rate)
		for worker in Workers:
			worker.set_model_parameters(aggregate_params)
			worker.new_learning_rate = new_learning_rate

		# Increase the step counter
		current_step += 1
tools.success("Finished...")