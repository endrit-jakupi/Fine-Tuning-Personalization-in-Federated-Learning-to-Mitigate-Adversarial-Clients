import random
from robust_aggregators import RobustAggregator

class Server_Fed(object):
    """Parameter server for distributed training."""
    def __init__(self, model_size, aggregator, second_aggregator, learning_rate, learning_rate_decay,
                 learning_rate_decay_delta, device, nb_workers, nb_byz, bucket_size, subsampling_ratio,
                 bit_precision, parameter_clamp):

        self.current_learning_rate = self.initial_learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_delta = learning_rate_decay_delta

        self.device = device
        self.nb_workers = nb_workers
        #JS: Number of workers to be subsampled by the server in every iteration
        self.subsampled_workers = int(subsampling_ratio * nb_workers)

        #JS: Instantiate the robust aggregator to be used
        self.robust_aggregator = RobustAggregator(aggregator, second_aggregator, self.subsampled_workers, nb_byz, bucket_size, model_size, device)

        #JS: security parameters (homomorphic encryption)
        self.bit_precision = bit_precision
        if bit_precision is not None and parameter_clamp is not None:
            self.quantization = (2**(bit_precision - 1) - 1) / parameter_clamp


    #JS: Compute the ids of the workers subsampled in every iteration
    def subsample_workers(self):
        if self.subsampled_workers < self.nb_workers:
            return random.sample(range(self.nb_workers), self.subsampled_workers)
        return range(self.nb_workers)


    #JS: Aggregate all incoming parameters
    def aggregate_parameters(self, worker_params):
        parameters = self.robust_aggregator.aggregate(worker_params)
        #JS: need to dequantize the aggregate parameters if encrypted
        if self.bit_precision is not None:
            parameters.mul_(1 / self.quantization)
        return parameters


    def update_learning_rate(self, current_step):
        if self.learning_rate_decay > 0 and current_step % self.learning_rate_decay_delta == 0:
            self.current_learning_rate = self.initial_learning_rate / (current_step / self.learning_rate_decay + 1)
        return self.current_learning_rate