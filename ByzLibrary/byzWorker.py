from byz_attacks import ByzantineAttack
from robust_aggregators import RobustAggregator

class ByzantineWorker(object):
    """A Byzantine worker for distributed training."""

    def __init__(self, nb_workers, nb_decl_byz, nb_real_byz, attack, aggregator, second_aggregator, server_clip, bucket_size,
                 model_size, mimic_learning_phase, gradient_clip, device):

        #JS: Instantiate the robust aggregator to be used (in particular to be used for auto ALIE and auto FOE)
        robust_aggregator = RobustAggregator(aggregator, second_aggregator, server_clip, nb_workers, nb_decl_byz, bucket_size, model_size, device)
        #JS: Instantiate the Byzantine attack to be used
        self.byzantine_attack = ByzantineAttack(attack, nb_real_byz, model_size, device, mimic_learning_phase, gradient_clip, robust_aggregator)

    def compute_byzantine_vectors(self, honest_vectors, grads_flipped, current_step):
        return self.byzantine_attack.generate_byzantine_vectors(honest_vectors, grads_flipped, current_step)

    def compute_byzantine_running_mean_and_var(self, honest_running_mean, honest_running_var, current_step):
        byzantine_dict_mean = dict()
        for batch_layer in honest_running_mean[0].keys():
            running_means = [honest_running_mean[worker][batch_layer] for worker in range(len(honest_running_mean))]
            byzantine_dict_mean[batch_layer] = self.byzantine_attack.generate_byzantine_vectors(running_means, None, current_step)

        byzantine_dict_var = dict()
        for batch_layer in honest_running_var[0].keys():
            running_vars = [honest_running_var[worker][batch_layer] for worker in range(len(honest_running_var))]
            byzantine_dict_var[batch_layer] =  self.byzantine_attack.generate_byzantine_vectors(running_vars, None, current_step)

        return byzantine_dict_mean, byzantine_dict_var