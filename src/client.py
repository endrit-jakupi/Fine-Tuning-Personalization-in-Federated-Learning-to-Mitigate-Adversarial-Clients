import torch

from src import setup_model
from torchvision import datasets, transforms
import sys

class Client():
	def __init__(self, config, client_id, dataloader, initial_weights = None, lbda = 0):
		'''
			config: dict
			id: int
			dataloader: tuple of train/test dataloaders
			initial_weights: list of tensors
		'''

		self.id = client_id

		self.device = config['device']

		self.training_dataloader = dataloader[0]
		self.testing_dataloader = dataloader[1]

		self.nb_classes = config['nb_classes']

		self.model = setup_model(config["model"], self.nb_classes  )
		# Logistic(self.nb_classes)

		if initial_weights is not None:
			self.set_model_parameters(initial_weights)

		self.model = self.model.to(self.device)

		self.model_size = len(torch.cat([param.view(-1) for param in self.model.parameters()]))

		self.criterion =torch.nn.CrossEntropyLoss().to(self.device) # torch.nn.BCEWithLogitsLoss().to(self.device)
		#torch.nn.CrossEntropyLoss().to(self.device)

		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'])

		self.lbda = lbda

		self.clip = None

	def step(self):
		self.optimizer.step()

	def get_id(self):
		return self.id

	def flatten(self, list_of_tensor):
		return torch.cat(tuple(tensor.view(-1) for tensor in list_of_tensor))

	def unflatten(self, flat_tensor, list_of_tensor):
		c = 0
		returned_list = [torch.zeros(tensor.shape) for tensor in list_of_tensor]
		for i, tensor in enumerate(list_of_tensor):
			count = torch.numel(tensor.data)
			returned_list[i].data = flat_tensor[c:c + count].view(returned_list[i].data.shape)
			c = c + count
		return returned_list
	def set_model_parameters(self, initial_weights):
		'''
			initial_weights: list of tensors
		'''
		for j, param in enumerate(self.model.parameters()):
			param.data = initial_weights[j].data.clone().detach()

	def set_model_gradient(self, flat_gradient):
		'''
			flat_gradient: flat tensor
		'''
		gradients = self.unflatten(flat_gradient, [param.grad for param in self.model.parameters()])

		for j, param in enumerate(self.model.parameters()):
			param.grad.data = gradients[j].data.clone().detach()

	def get_model_parameters(self):
		return [param for param in self.model.parameters()]
	def set_model_parameters_with_flat_tensor(self, flat_tensor):
		'''
			initial_weights: flat tensor
		'''
		list_of_parameters = self.unflatten(flat_tensor, self.get_model_parameters())
		for j, param in enumerate(self.model.parameters()):
			param.data = list_of_parameters[j].data.clone().detach()
		
	def get_flatten_model_parameters(self):
		return self.flatten([param for param in self.model.parameters()])


	def set_model_gradient_with_flat_tensor(self, flat_gradient):
		'''
			flat_gradient: flat tensor
		'''
		self.optimizer.zero_grad()

		gradients = self.unflatten(flat_gradient, [param for param in self.model.parameters()])
		for j, param in enumerate(self.model.parameters()):
			param.grad = gradients[j].clone().detach()
			
	def compute_gradient(self):
		self.model.train()
		self.model.zero_grad()
		inputs, targets = next(iter(self.training_dataloader))
		inputs, targets = inputs.to(self.device), targets.to(self.device)
		
		outputs = self.model(inputs)
		loss = self.criterion(outputs, targets.to(torch.long))
		loss.backward()

		grad = self.flatten([param.grad.data for param in self.model.parameters()]) + self.lbda * self.flatten([param.data for param in self.model.parameters()])

		return self.clip_gradient(grad), loss

	def compute_full_gradient(self):
		self.model.train()
		self.model.zero_grad()

		# print(len(inputs))
		outputs = self.model(inputs)

		loss = self.criterion(outputs, targets)

		loss.backward()

		grad = self.flatten([param.grad.data for param in self.model.parameters()]) + self.lbda * self.flatten([param.data for param in self.model.parameters()])

		return self.clip_gradient(grad), loss
	

	def clip_gradient(self, gradient):
		if self.clip is None:
			return gradient

		grad_norm = gradient.norm().item()
		if grad_norm > self.clip:
			gradient.mul_(self.clip / grad_norm)
		return gradient

	def evaluate(self):
		with torch.no_grad():
			total = 0
			correct = 0 
			for data in self.testing_dataloader:
				inputs, targets = data
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += targets.size(0)
				correct += (predicted == targets).sum().item()
				

		return 100*(correct/total)

	"""def evaluate_balanced_accuracy(self):
		# Balanced accuracy is the average of sensitivity and specificity
  
		with torch.no_grad():
			# """

	def evaluate_balanced_accuracy(self):
		with torch.no_grad():
			true_positives = 0
			true_negatives = 0
			false_positives = 0
			false_negatives = 0

			for data in self.testing_dataloader:
				inputs, targets = data
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.model(inputs)
				_, predicted = torch.max(outputs.data, 1)

				true_positives += ((predicted == 1) & (targets == 1)).sum().item()
				true_negatives += ((predicted == 0) & (targets == 0)).sum().item()
				false_positives += ((predicted == 1) & (targets == 0)).sum().item()
				false_negatives += ((predicted == 0) & (targets == 1)).sum().item()

			sensitivity = true_positives / (true_positives + false_negatives)
			specificity = true_negatives / (true_negatives + false_positives)
			balanced_accuracy = (sensitivity + specificity) / 2

		return balanced_accuracy
	
	
	def evaluate_f1_score(self):
		with torch.no_grad():
			true_positives = 0
			false_positives = 0
			false_negatives = 0
			for data in self.testing_dataloader:
				inputs, targets = data
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				true_positives += ((predicted == 1) & (targets == 1)).sum().item()
				false_positives += ((predicted == 1) & (targets == 0)).sum().item()
				false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
			
			f1_score = 2 * true_positives / (2 * true_positives + false_positives + false_negatives + 1e-7)
			
		return f1_score
		
	def evaluate_loss(self):
		with torch.no_grad():
			total = 0
			loss = 0 
			for data in self.testing_dataloader:
				inputs, targets = data
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.model(inputs)
				loss += self.criterion(outputs, targets.to(torch.long)).item()
				total += targets.size(0)
		return loss/total