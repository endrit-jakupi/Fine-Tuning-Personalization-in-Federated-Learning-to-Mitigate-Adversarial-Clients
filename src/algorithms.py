
from src.client import Client
from src.server import Server
import numpy as np
from tqdm import tqdm, trange
import sys
sys.path.append('ByzLibrary')
import torch
from robust_aggregators import RobustAggregator
from byz_attacks import ByzantineAttack


# This file contains personalized Federated Learning algorithms 

class PersonalizedAlgorithm(object):
    def __init__(self, config, dataloaders, model_size, return_loss=False):
        self.config = config
        self.dataloaders = dataloaders
        self.model_size = model_size
        self.aggregation_rule = RobustAggregator('nnm', config['agg'], 1, config['n'], config['f'], 0, model_size, config['device'])
        self.attack = ByzantineAttack(config['attack'], config['f'], model_size, config['device'], learning_phase = 0, gradient_clip = 1, robust_aggregator = self.aggregation_rule)
        self.return_loss = return_loss

    def run(self):
        # Execute personalized learning algorithm
        # Return the results
        pass




class FedAvg(PersonalizedAlgorithm):
    def __init__(self, config, dataloaders, model_size, return_loss=False):
        super().__init__(config, dataloaders, model_size, return_loss)

    def run(self):
        config = self.config
        nb_honest= config['n'] - config['f']
        results_array = np.zeros((nb_honest))
        losses_array = np.zeros((config['T'], nb_honest))
        server = Server(config)
        honest_clients = [Client(config, i, self.dataloaders[i]) for i in range(nb_honest)]
        for epoch in trange(config['T']):
            params = []
            for client in honest_clients:
                client.set_model_parameters(server.get_model_parameters())
                grad, loss = client.compute_gradient()
                losses_array[epoch, client.id] = loss
                params.append(grad)
            byz_params = self.attack.generate_byzantine_vectors(params, None, epoch)
            params = params + byz_params
            fed_avr_param = self.aggregation_rule.aggregate(params)
            server.set_model_gradient_with_flat_tensor(fed_avr_param)
            server.step()

        for i in range(nb_honest):
            results_array[i] = honest_clients[i].evaluate()

        if self.return_loss:
            return results_array, losses_array
        else:
            return results_array

class IPGD(PersonalizedAlgorithm):
    """
    Interpolated Personalized Gradient Descent
    This algorithm return only the accuracy results of a selection of clients (nb_main_client)
    """
    def __init__(self, config, dataloaders, model_size, return_loss=False, lams=[0, 0.2, 0.4, 0.6, 0.8, 1], eval_every= 10): 
        super().__init__(config, dataloaders, model_size, return_loss)
        self.lams = lams
        self.nb_main_client = config["nb_main_client"]
        self.eval_every = eval_every

    def run(self):
        config = self.config
        nb_honest= config['n'] - config['f']
        results_array = np.zeros((len(self.lams), self.nb_main_client))
        results_array_checkpoints = np.zeros((len(self.lams), self.nb_main_client, config['T']//self.eval_every))
        f1_score = np.zeros((len(self.lams), self.nb_main_client))
        f1_score_checkpoints = np.zeros((len(self.lams), self.nb_main_client, config['T']//self.eval_every))
        losses_array = np.zeros((config['T'], len(self.lams), self.nb_main_client))
        for lam in self.lams:
            honest_clients = [Client(config, i, self.dataloaders[i]) for i in range(nb_honest)]
            for j in range(self.nb_main_client): 
                main_client = honest_clients[j]
                for epoch in trange(config['T']): 

                    params = []
                    losses = []
                    for other_client in honest_clients:
                        other_client_model = other_client.get_model_parameters()
                        other_client.set_model_parameters(main_client.get_model_parameters())
                        grad, loss = other_client.compute_gradient()
                        losses.append(loss)
                        params.append(grad)
                        other_client.set_model_parameters(other_client_model)
                    
                    byz_params = self.attack.generate_byzantine_vectors(params, None, epoch)
                    params = params + byz_params

                    avg_loss = torch.mean(torch.tensor(losses))
                    #params_cpu = [param.to('cpu') for param in params] #
                    
                    #fed_avr_param = aggregation_rule.aggregate(params_cpu) #
                
                    #fed_avr_param = fed_avr_param.to(main_client.device) #

                    fed_avr_param = self.aggregation_rule.aggregate(params)

                    grad, loss = main_client.compute_gradient()
                    losses_array[epoch, self.lams.index(lam), main_client.id] = (1-lam)*loss + lam * avg_loss
                    #losses_array[run, config['lams'].index(lam), epoch] = loss
                    params = (1-lam)* grad + lam * fed_avr_param

                    main_client.set_model_gradient_with_flat_tensor(params)
                    main_client.step()
                    if (epoch+1) % self.eval_every == 0:
                        acc = main_client.evaluate()
                        f1 = main_client.evaluate_f1_score()
                        results_array_checkpoints[ self.lams.index(lam), j, epoch//self.eval_every] = acc
                        f1_score_checkpoints[ self.lams.index(lam), j, epoch//self.eval_every] = f1

                results_array[ self.lams.index(lam), j] = main_client.evaluate()
                f1_score[ self.lams.index(lam), j] = main_client.evaluate_f1_score()
        if self.return_loss:
            return results_array,f1_score, losses_array, results_array_checkpoints, f1_score_checkpoints
        else:
            return results_array,f1_score
        


class IPSGD(PersonalizedAlgorithm):
    """
    Interpolated Personalized Stochastic Gradient Descent
    This algorithm return only the accuracy results of a selection of clients (nb_main_client)
    """
    def __init__(self, config, dataloaders, model_size, return_loss=False, lams=[0,0.2,0.4,0.6,0.8,1]):
        super().__init__(config, dataloaders, model_size, return_loss)
        self.lams = lams
        self.nb_main_client = config["nb_main_client"]

    def run(self):
        config = self.config
        nb_honest= config['n'] - config['f']
        results_array = np.zeros((len(self.lams), self.nb_main_client))
        losses_array = np.zeros((config['T'], len(self.lams), self.nb_main_client))
        for lam in self.lams:
            honest_clients = [Client(config, i, self.dataloaders[i]) for i in range(nb_honest)]
            for j in range(self.nb_main_client): 
                main_client = honest_clients[j]
                for epoch in trange(config['T']): 

                    params = []
                    losses = []
                    for other_client in honest_clients:
                        other_client_model = other_client.get_model_parameters()
                        other_client.set_model_parameters(main_client.get_model_parameters())
                        grad, loss = other_client.compute_gradient()
                        losses.append(loss)
                        params.append(grad)
                        other_client.set_model_parameters(other_client_model)
                    
                    byz_params = self.attack.generate_byzantine_vectors(params, None, epoch)
                    params = params + byz_params

                    avg_loss = torch.mean(torch.tensor(losses))
                    #params_cpu = [param.to('cpu') for param in params] #
                    
                    #fed_avr_param = aggregation_rule.aggregate(params_cpu) #
                
                    #fed_avr_param = fed_avr_param.to(main_client.device) #

                    fed_avr_param = self.aggregation_rule.aggregate(params)

                    grad, loss = main_client.compute_gradient()
                    losses_array[epoch, self.lams.index(lam), main_client.id] = (1-lam)*loss + lam * avg_loss
                    #losses_array[run, config['lams'].index(lam), epoch] = loss
                    params = (1-lam)* grad + lam * fed_avr_param

                    main_client.set_model_gradient_with_flat_tensor(params)
                    main_client.step()

                results_array[ self.lams.index(lam), j] = main_client.evaluate()
        if self.return_loss:
            return results_array, losses_array
        else:
            return results_array
        



# TODO add return loss
class E_IPGD(PersonalizedAlgorithm): # TODO to be verified
    """
    Interpolated Personalized Gradient Descent
    This algorithm return only the accuracy results of a selection of clients (nb_main_client)
    """
    def __init__(self, config, dataloaders, model_size, lams=[0,0.2,0.4,0.6,0.8,1], nb_main_client = 3):
        super().__init__(config, dataloaders, model_size)
        self.lams = lams
        self.nb_main_client = nb_main_client

    def run(self):
        config = self.config
        nb_honest= config['n'] - config['f']
        results_array = np.zeros((len(self.lams), self.nb_main_client))
        
        for lam in self.lams:
            server = Server(config)
            honest_clients = [Client(config, i, self.dataloaders[i]) for i in range(nb_honest)]
            for epoch in trange(config['T']): 
                params = []
                for client in honest_clients:
                    client_model = client.get_model_parameters()
                    client.set_model_parameters(server.get_model_parameters())
                    grad, loss = client.compute_gradient()
                    params.append(grad)
                
                    client.set_model_parameters(client_model)
                
                byz_params = self.attack.generate_byzantine_vectors(params, None, epoch)
                params_with_corruption = params + byz_params

                fed_avr_param = self.aggregation_rule.aggregate(params_with_corruption)
                
                server.set_model_gradient_with_flat_tensor(fed_avr_param)
                server.step()

                for i, client in enumerate(honest_clients):
                    grad, loss = client.compute_gradient()
                    #losses_array[run, config['lams'].index(lam), epoch, i] = loss
                    client.set_model_gradient_with_flat_tensor(grad + (1-lam) * (fed_avr_param-params[i])) # Option 1 
                    #client.set_model_gradient_with_flat_tensor(grad + lam * (fed_avr_param-grad)) # Option 2 
                    client.step()
            for j in range(self.nb_main_client):
                results_array[ self.lams.index(lam), j] = honest_clients[j].evaluate()
        return results_array

class ModelInterpolation(PersonalizedAlgorithm):
    """
    Model Interpolation Federated Learning Algorithm
    """
    def __init__(self, config, dataloaders, model_size, lams=[0,0.2,0.4,0.6,0.8,1], nb_main_client = 1):
        super().__init__(config, dataloaders, model_size)
        self.lams = lams
        self.nb_main_client = nb_main_client

    def run(self):
        config = self.config
        nb_honest= config['n'] - config['f']
        results_array = np.zeros((len(self.lams), self.nb_main_client))
        
        for lam in self.lams:
            server = Server(config)
            honest_clients = [Client(config, i, self.dataloaders[i]) for i in range(nb_honest)]
            for epoch in trange(config['T']): 
                params = []
                for client in honest_clients:
                    client_model = client.get_model_parameters()
                    client.set_model_parameters(server.get_model_parameters())
                    grad, loss = client.compute_gradient()
                    params.append(grad)
                
                    client.set_model_parameters(client_model)
                
                byz_params = self.attack.generate_byzantine_vectors(params, None, epoch)
                params_with_corruption = params + byz_params

                fed_avr_param = self.aggregation_rule.aggregate(params_with_corruption)
                
                server.set_model_gradient_with_flat_tensor(fed_avr_param)
                server.step()

                for i, client in enumerate(honest_clients):
                    grad, loss = client.compute_gradient()
                    #losses_array[run, config['lams'].index(lam), epoch, i] = loss
                    client.set_model_gradient_with_flat_tensor(grad + (1-lam) * (fed_avr_param-params[i])) # Option 1 
                    #client.set_model_gradient_with_flat_tensor(grad + lam * (fed_avr_param-grad)) # Option 2 
                    client.step()
            for j in range(self.nb_main_client):
                results_array[ self.lams.index(lam), j] = honest_clients[j].evaluate()
        return results_array



class SplitParameters(PersonalizedAlgorithm):
    """
    Split Parameters Federated Learning Algorithm
    """

    def __init__(self, config, dataloaders, model_size, split_sizes=[0, 0.25, 0.5, 0.75, 1]):
        super().__init__(config, dataloaders, model_size)
        self.split_sizes = split_sizes

    def run(self):
        config = self.config
         
        nb_honest = config['n'] - config['f']
        results_array = np.zeros((len(self.split_sizes), nb_honest))
        for j, split_size in enumerate(self.split_sizes):
            honest_clients = [Client(config, i, self.dataloaders[i]) for i in range(nb_honest)]
            server = Server(config)
            for epoch in trange(config['T']): 

                params_omega = []
                grad_omega_server = server.get_flatten_model_parameters()[:int(split_size*self.model_size)]
                for client in honest_clients:
                    client_flat_model = client.get_flatten_model_parameters().clone()
                    client_flat_model[:int(split_size*len(client_flat_model))] = grad_omega_server # setteing client omega to the common part of the model
                    client.set_model_parameters_with_flat_tensor(client_flat_model)
                    
                    grad, loss = client.compute_gradient()
                    params_omega.append(grad[:int(split_size*len(grad))])
                    grad_theta = torch.zeros(len(grad)).to(client.device)
                    grad_theta[int(split_size*len(grad)):] = grad[int(split_size*len(grad)):]
                    client.set_model_gradient_with_flat_tensor(grad_theta)
                    client.step()
                if epoch ==0:
                    print('length of shared params ', len(params_omega[0]))
                byz_params = self.attack.generate_byzantine_vectors(params_omega, None, epoch)
                params_with_corruption = params_omega + byz_params

                fed_avr_param = self.aggregation_rule.aggregate(params_with_corruption)

                grad_omega = torch.zeros(self.model_size).to(server.device)
                grad_omega[:len(fed_avr_param)] = fed_avr_param
                server.set_model_gradient_with_flat_tensor(grad_omega)
                server.step()
        
            for i, client in enumerate(honest_clients):
                #losses_array[run, config['split_size'].index(split_size), :, i] = client.losses
                results_array[j,i] = client.evaluate()
                print("split size", split_size, "client", i, "accuracy", results_array[j,i])

        return results_array


algorithms_dict = {"FedAvg": FedAvg, "IPGD": IPGD, "E_IPGD": E_IPGD, "SplitParameters": SplitParameters}