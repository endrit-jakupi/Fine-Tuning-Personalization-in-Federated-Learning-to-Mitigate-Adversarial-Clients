import torch, random
from robust_aggregators import RobustAggregator
import models, misc

class P2PWorker(object):
    """A worker for decentralized learning in peer to peer model."""

    def __init__(self, worker_id, data_loader, data_loader_test, nb_workers, nb_byz, aggregator, second_aggregator, bucket_size, model,
                 learning_rate, learning_rate_decay, learning_rate_decay_delta, weight_decay, loss, momentum, device, labelflipping,
                 gradient_clip, numb_labels):
        self.worker_id = worker_id
        self.nb_byz = nb_byz
        self.nb_honest = nb_workers - nb_byz

        self.loaders = {"train": data_loader, "test": data_loader_test}
        self.iterators = {"train": iter(data_loader), "test": iter(data_loader_test)}

        self.initial_learning_rate = self.current_learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_delta = learning_rate_decay_delta

        self.device = device
        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)()
        self.model.to(self.device)
        #JS: list of shapes of the model in question. Used when unflattening gradients and model parameters
        self.model_shapes = [param.shape for param in self.model.parameters()]
        self.model_size = len(misc.flatten(self.model.parameters()))

        if self.device == "cuda":
            #JS: model is on GPU and not explicitly restricted to one particular card => enable data parallelism
            self.model = torch.nn.DataParallel(self.model, device_ids = [0, 1])
        #self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay)
        
        #JS: Instantiate the robust aggregator to be used
        self.robust_aggregator = RobustAggregator(aggregator, second_aggregator, self.nb_honest, nb_byz, bucket_size, self.model_size, self.device)
        
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.momentum = momentum
        self.gradient_clip = gradient_clip

        self.labelflipping = labelflipping
        self.numb_labels = numb_labels

    #JS: Sample train or test batch, depending on the mode
    #JS: mode can be "train" or "test"
    def sample_batch(self, mode):
        try:
            return next(self.iterators[mode])
        except:
            self.iterators[mode] = iter(self.loaders[mode])
            return next(self.iterators[mode])


    #JS: Generic function to compute gradient on batch = (inputs, targets)
    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return misc.flatten([param.grad for param in self.model.parameters()])


    #JS: Compute honest gradient and flipped gradient (if required)
    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        #JS: compute gradient on flipped labels
        if self.labelflipping:
            #JS: in case batch norm is used, set the model in eval mode when computing the flipped gradient,
            # in order not to change the running mean and variance
            self.model.eval()
            targets_flipped = targets.sub(self.numb_labels - 1).mul(-1)
            self.gradient_labelflipping = self.backward_pass(inputs, targets_flipped)
            self.model.train()

        #JS: compute honest gradient (i.e., on current parameters and on true labels)
        return self.backward_pass(inputs, targets)


    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.compute_gradients(), alpha=1-self.momentum)

        if self.gradient_clip is not None:
            return misc.clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient


    def local_model_update(self, current_step):
        def update_learning_rate(step):
            if self.learning_rate_decay > 0 and step % self.learning_rate_decay_delta == 0:
                return self.initial_learning_rate / (step / self.learning_rate_decay + 1)
            else:
                return self.current_learning_rate

        #JS: Update the learning rate
        new_learning_rate = update_learning_rate(current_step)
        if self.current_learning_rate != new_learning_rate:
            self.current_learning_rate = new_learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_learning_rate

        # Perform the update step
        self.optimizer.step()


    def perform_local_step(self, current_step):
        self.set_gradient(self.compute_momentum())
        self.local_model_update(current_step)
        return misc.flatten(self.model.parameters())


    def aggregate_and_update_parameters(self, honest_params, byzantine_params):
        #JS: Generate randomly n-2f-1 indices in [0, ..., worker_id-1, worker_id+1, ..., n-f-1],
        #corresponding to the honest workers who "respond" to worker worker_id (in addition to all Byzantine workers
        #and worker worker_id himself)
        indices_list = [x for x in range(self.nb_honest) if x != self.worker_id]
        random_indices = random.sample(indices_list, self.nb_honest - self.nb_byz - 1)

        #JS: worker_params is a list concatenating the parameter vectors of the n-2f-1 honest workers that "respond" and the Byzantine vectors
        worker_params = [honest_params[k] for k in random_indices] + byzantine_params

        #JS: append the parameter vector of worker in question (i.e., worker worker_id) at the end of the list
        worker_params.append(honest_params[self.worker_id])

        #JS: Aggregate all incoming parameters
        aggregate_params = self.robust_aggregator.aggregate(worker_params)

        # Update the model parameters
        self.set_model_parameters(aggregate_params)


    @torch.no_grad()
    def compute_accuracy(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs, targets in self.loaders["test"]:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct/total


    def set_gradient(self, gradient):
        """ Overwrite the gradient with the given one."""
        gradient = misc.unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].detach().clone()


    def set_model_parameters(self, params):
        params = misc.unflatten(params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = params[j].data.detach().clone()