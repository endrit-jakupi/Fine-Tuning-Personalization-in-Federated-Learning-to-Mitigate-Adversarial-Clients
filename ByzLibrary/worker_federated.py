import torch
import models, misc

class Worker_Fed(object):
    """A worker for federated learning."""
    def __init__(self, data_loader, test_loader, local_steps, model, learning_rate, weight_decay, loss, momentum,
                 gradient_clip, bit_precision, parameter_clamp, device):

        self.data_loader = data_loader
        self.test_loader = test_loader
        self.train_iterator = iter(data_loader)

        self.device = device
        self.local_steps = local_steps
        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)()
        self.model.to(self.device)
        #JS: list of shapes of the model in question. Used when unflattening gradients and model parameters
        self.model_shapes = [param.shape for param in self.model.parameters()]
        self.model_size = len(misc.flatten(self.model.parameters()))
        if self.device == "cuda":
            #JS: model is on GPU and not explicitly restricted to one particular card => enable data parallelism
            self.model = torch.nn.DataParallel(self.model, device_ids = [0, 1])

        self.current_learning_rate = self.new_learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.momentum = momentum
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.gradient_clip = gradient_clip

        #JS: security parameters (homomorphic encryption)
        self.bit_precision, self.parameter_clamp = bit_precision, parameter_clamp
        if bit_precision is not None and parameter_clamp is not None:
            self.quantization = (2**(self.bit_precision - 1) - 1) / self.parameter_clamp


    def sample_train_batch(self):
        try:
            return next(self.train_iterator)
        except:
            self.train_iterator = iter(self.data_loader)
            return next(self.train_iterator)


    #JS: Compute gradient (on current parameters)
    def backward_pass(self):
        self.model.train()
        inputs, targets = self.sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        loss = self.loss(self.model(inputs), targets)
        self.model.zero_grad()
        loss.backward()
        unflattened_grad = [param.grad for param in self.model.parameters()]
        return misc.flatten(unflattened_grad)


    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.backward_pass(), alpha=1-self.momentum)

        if self.gradient_clip is not None:
            return misc.clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient


    #JS: Clamp the parameters and quantize them in case of encryption
    def quantize_parameters(self, parameters):
        clamped_params = torch.clamp(parameters, min=-self.parameter_clamp, max=self.parameter_clamp)
        if self.bit_precision is not None:
            torch.round_(clamped_params.mul_(self.quantization))
        return clamped_params


    def set_model_parameters(self, server_params):
        """ Overwrite the model parameters with the ones sent by the server."""
        server_params = misc.unflatten(server_params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = server_params[j].data.clone().detach()


    def set_gradient(self, gradient):
        """ Overwrite the gradient with the given one."""
        gradient = misc.unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].clone().detach()


    def local_model_update(self):
        #JS: Update the learning rate (if changed)
        if self.current_learning_rate != self.new_learning_rate:
            self.current_learning_rate = self.new_learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.new_learning_rate

        # Perform the update step
        self.optimizer.step()


    def update_parameters(self):
        for _ in range(self.local_steps):
            self.set_gradient(self.compute_momentum())
            self.local_model_update()

        if self.parameter_clamp is not None:
            #JS: Need to clamp the parameters (and potentially quantize them for encryption)
            return self.quantize_parameters(misc.flatten(self.model.parameters()))
        return misc.flatten(self.model.parameters())


    @torch.no_grad()
    def compute_accuracy(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct/total