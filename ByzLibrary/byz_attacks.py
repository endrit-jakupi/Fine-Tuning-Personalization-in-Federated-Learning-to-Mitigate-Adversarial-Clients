import torch, misc, copy

def signflipping(honest_vectors, **kwargs):
    return torch.stack(honest_vectors).mean(dim=0).mul(-1)


def labelflipping(flipped_vectors, **kwargs):
    return torch.stack(flipped_vectors).mean(dim=0)


def fall_of_empires(honest_vectors, attack_factor=3, negative=False, **kwargs):
    #JS: negative controls the sign of the attack_factor
    #JS: attack_factor controls the magnitude of the attack_vector to be added to avg_honest_vector
    if negative:
        attack_factor = - attack_factor
    return torch.stack(honest_vectors).mean(dim=0).mul(1 - attack_factor)


def auto_FOE(honest_vectors, aggregator, nb_byz, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_FOE(factor):
        temp_vectors = copy.deepcopy(honest_vectors)
        byzantine_vectors = [fall_of_empires(temp_vectors, attack_factor=factor)] * nb_byz
        distance = aggregator.aggregate(temp_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = misc.line_maximize(eval_factor_FOE)
    return fall_of_empires(honest_vectors, attack_factor=best_factor)


def a_little_is_enough(honest_vectors, attack_factor=3, negative=False, **kwargs):
    #JS: negative controls the sign of the attack_factor
    #JS: attack_factor controls the magnitude of the attack_vector to be added to avg_honest_vector
    stacked_vectors = torch.stack(honest_vectors)
    attack_vector = stacked_vectors.var(dim=0).sqrt_()
    if negative:
        attack_factor = - attack_factor
    return stacked_vectors.mean(dim=0).add(attack_vector, alpha=attack_factor)


def auto_ALIE(honest_vectors, aggregator, nb_byz, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_ALIE(factor):
        temp_vectors = copy.deepcopy(honest_vectors)
        byzantine_vectors = [a_little_is_enough(temp_vectors, attack_factor=factor)] * nb_byz
        distance = aggregator.aggregate(temp_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = misc.line_maximize(eval_factor_ALIE)
    return a_little_is_enough(honest_vectors, attack_factor=best_factor)


def mimic(attack, honest_vectors, current_step, **kwargs):
    if attack.learning_phase_mimic is None:
        #JS: Always return the vector of the first worker
        return honest_vectors[0]

    if current_step < attack.learning_phase_mimic:
        #JS: Update mu_mimic and z_mimic only if still in learning phase
        attack.update_mimic_heuristic(honest_vectors, current_step)
        #JS: Return the vector of the first worker in learning phase
        return honest_vectors[0]

    current_max = -1
    best_worker_to_mimic = None
    for i, vector in enumerate(honest_vectors):
        dot_product = torch.dot(vector, attack.z_mimic).norm().item()
        if dot_product > current_max:
            current_max = dot_product
            best_worker_to_mimic = i

    return honest_vectors[best_worker_to_mimic]


#JS: infinity attack
def inf(attack, **kwargs):
    return torch.ones(attack.model_size, device=attack.device) * float('inf')

# Gradient Reversion Attack
def gradient_reversion(honest_vectors, scale=5, **kwargs):
    """
    Reverses and scales the average gradient from honest clients.

    Args:
        honest_vectors: gradients from honest clients
        scale: factor to multiply after reversing (default 5)

    Returns:
        A malicious (reversed) gradient tensor
    """
    return torch.stack(honest_vectors).mean(dim=0).mul(-scale)

# Partial Reversion Attack
def partial_reversion(honest_vectors, scale=1000, portion=0.00001, **kwargs):
    """
    Reverses only a small portion of the gradient values.

    Args:
        honest_vectors: gradients from honest clients
        scale: multiplier for flipped values (default 1000)
        portion: fraction of total gradient elements to flip (default 0.001%)

    Returns:
        A poisoned gradient tensor with partial inversion
    """
    stacked = torch.stack(honest_vectors)
    avg_grad = stacked.mean(dim=0)

    # Flatten for manipulation
    flat = avg_grad.flatten()
    num_total = flat.numel()
    num_to_flip = int(num_total * portion)

    # Flip the first `num_to_flip` values
    flat[:num_to_flip] *= -scale

    # Reshape back to original
    return flat.view_as(avg_grad)

# Gaussian Noise Attack
def gaussian_noise_attack(honest_vectors, noise_scale=0.1, **kwargs):
    """
    Adds Gaussian noise to the average gradient to create a noisy poisoned update.

    Args:
        honest_vectors: list of gradient tensors from honest clients
        noise_scale: standard deviation of the Gaussian noise (default: 0.1)

    Returns:
        A noisy poisoned gradient tensor
    """
    avg_grad = torch.stack(honest_vectors).mean(dim=0)
    noise = torch.randn_like(avg_grad) * noise_scale
    return avg_grad + noise


#JS: Dictionary mapping every Byzantine attack to its corresponding function
byzantine_attacks = {"SF": signflipping, "LF": labelflipping, "FOE": fall_of_empires, "ALIE": a_little_is_enough, "mimic": mimic,
                    "auto_ALIE": auto_ALIE, "auto_FOE": auto_FOE, "inf": inf, "GR": gradient_reversion, "PR": partial_reversion, "NS": gaussian_noise_attack}

class ByzantineAttack(object):

    def __init__(self, attack_name, nb_real_byz, model_size, device, learning_phase, gradient_clip, robust_aggregator):
        self.attack_name = attack_name
        #JS: nb_real_byz is the actual number of "real" Byzantine workers
        self.nb_real_byz = nb_real_byz
        #JS: Instantiate the robust aggregator to be used (in particular to be used for auto ALIE and auto FOE)
        self.robust_aggregator = robust_aggregator
        self.model_size = model_size
        self.gradient_clip = gradient_clip
        self.device = device
        #JS: parameters for the mimic_heuristic attack
        self.z_mimic = torch.rand(model_size, device=device)
        self.mu_mimic = torch.zeros(model_size, device=device)
        self.learning_phase_mimic = learning_phase


    def generate_byzantine_vectors(self, honest_vectors, flipped_vectors, current_step):
        if self.nb_real_byz == 0:
            return list()

        #JS: Compute the Byzantine vectors
        byzantine_vector = byzantine_attacks[self.attack_name](attack=self, honest_vectors=honest_vectors, flipped_vectors=flipped_vectors, learning_phase=self.learning_phase_mimic,
                                                   current_step=current_step, aggregator=self.robust_aggregator, nb_byz = self.nb_real_byz)
        if self.gradient_clip is not None:
            byzantine_vector = misc.clip_vector(byzantine_vector, self.gradient_clip)

        return [byzantine_vector] * self.nb_real_byz


    #JS: update the parameters of the mimic attack
    def update_mimic_heuristic(self, honest_vectors, current_step):
        time_factor = 1 / (current_step + 2)
        step_ratio = (current_step + 1) * time_factor
        self.mu_mimic.mul_(step_ratio)
        self.mu_mimic.add_(torch.stack(honest_vectors).mean(dim=0), alpha=time_factor)

        self.z_mimic.mul_(step_ratio)
        cumulative = torch.zeros(self.model_size, device=self.device)
        for vector in honest_vectors:
            deviation = torch.sub(vector, self.mu_mimic)
            deviation.mul_(torch.dot(deviation, self.z_mimic).norm().item())
            cumulative.add_(deviation)

        self.z_mimic.mul_(step_ratio)
        self.z_mimic.add_(torch.nn.functional.normalize(cumulative, dim=0), alpha=time_factor)
        self.z_mimic.div_(self.z_mimic.norm().item())
