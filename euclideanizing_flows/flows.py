import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import numpy as np

class NaturalGradientDescentVelNet(nn.Module):
	"""
	taskmap_fcn: map to a latent space
	grad_taskmap_fcn: jacobian of the map
	grad_potential_fcn: gradient of a potential fcn defined on the mapped space
	n_dim_x: observed (input) space dimensions
	n_dim_y: latent (output) space dimentions
	origin (optional): shifted origin of the input space (this is the goal usually)
	scale_vel (optional): if set to true, learns a scalar velocity multiplier
	is_diffeomorphism (optional): if set to True, use the inverse of the jacobian itself rather than pseudo-inverse
	"""
	def __init__(self, taskmap_fcn, grad_potential_fcn, n_dim_x, n_dim_y,
				 scale_vel=True,
				 origin=None, eps=1e-12, device='cpu'):

		super(NaturalGradientDescentVelNet, self).__init__()
		self.taskmap_fcn = taskmap_fcn
		self.grad_potential_fcn = grad_potential_fcn
		self.n_dim_x = n_dim_x
		self.n_dim_y = n_dim_y
		self.eps = eps
		self.device = device
		self.I = torch.eye(self.n_dim_x, self.n_dim_x, device=device).unsqueeze(0)
		self.scale_vel = scale_vel
		self.vv=nn.Sequential(nn.Linear(2,300),
							  nn.PReLU(),
							  nn.Linear(300,300),
							  nn.PReLU(),
							  nn.Linear(300,2))

		# scaling network (only used when scale_vel param is True!)
		self.log_vel_scalar = CouplingLayer(n_dim_x, 1, 100, act='leaky_relu')					 # a 2-hidden layer network
		self.vel_scalar = lambda x: torch.exp(self.log_vel_scalar(x)) + self.eps 		 # always positive scalar!

		if origin is None:
			self.origin = torch.zeros(1, self.n_dim_x, device=self.device)
		else:
			self.origin = origin.to(device)



	def forward(self, x):
		if x.ndimension() == 1:
			flatten_output = True  # output the same dimensions as input
			x = x.view(1, -1)
		else:
			flatten_output = False

		origin_, _ = self.taskmap_fcn(self.origin)
		y_hat, J_hat = self.taskmap_fcn(x)
		y_hat = y_hat - origin_  			# Shifting the origin
		doty_pre=self.vv(y_hat)
		yy=y_hat

		ls_var = torch.sum(doty_pre * yy, -1, keepdim=True)
		V_y = torch.sum(yy * yy, -1, keepdim=True)
		yd_hat = doty_pre - F.relu(ls_var + 1e-4 * V_y) * yy / (V_y + 1e-12)
		#	yd_hat = -self.grad_potential_fcn(y_hat)  		# negative gradient of potential


		J_hat_inv = torch.linalg.pinv(J_hat)


		xd_hat = torch.bmm(J_hat_inv, F.normalize(yd_hat).unsqueeze(2)).squeeze() 	# natural gradient descent

		if self.scale_vel:
			xd = self.vel_scalar(x) * xd_hat  							# mutiplying with a learned velocity scalar
		else:
			xd = xd_hat

		if flatten_output:
			xd = xd.squeeze()

		return xd


class BijectionNet(nn.Sequential):
	"""
	A sequential container of flows based on coupling layers.
	"""
	def __init__(self, num_dims, num_blocks, num_hidden, act=None,is_diffeomorphism=False):
		self.is_diffeomorphism = is_diffeomorphism

		self.num_dims = num_dims
		modules = []
		# mask = mask.to(device).float()
		for _ in range(num_blocks):
			modules += [
				CouplingLayer(
					num_inputs=num_dims, num_outputs=num_dims,num_hidden=num_hidden,
					act=act),
			]
		super(BijectionNet, self).__init__(*modules)

	def jacobian(self, inputs):
		'''
		Finds the product of all jacobians
		'''
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		for module in self._modules.values():
			J_module = module.jacobian(inputs)
			J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)

		return J

	def forward(self, inputs):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		for module in self._modules.values():
			J_module = module.jacobian(inputs)
			J = torch.matmul(J_module, J)
			inputs = module(inputs,is_diffeomorphism=self.is_diffeomorphism)

		return inputs, J


class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, num_outputs,num_hidden, act='elu'):
		super(CouplingLayer, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus,'silu':nn.SiLU}

		act_func = activations[act]

		self.nn1 = nn.Sequential(
			torch.nn.utils.parametrizations.spectral_norm(nn.Linear(num_inputs,num_hidden)),
			nn.Tanh(),
			torch.nn.utils.parametrizations.spectral_norm(nn.Linear(num_hidden, num_hidden)),
			nn.Tanh(),

			torch.nn.utils.parametrizations.spectral_norm(nn.Linear(num_hidden,num_outputs)),

		)
		self.nn2_1 = nn.Sequential(
			nn.Linear(num_inputs,num_hidden),
			act_func(),
			nn.Linear(num_hidden,num_hidden),
		    act_func(),
			nn.Linear(num_hidden,num_outputs),
			nn.Softplus()
		)



		self.nn3 = nn.Sequential(
			nn.Linear(num_inputs, num_hidden),
			act_func(),
			nn.Linear(num_hidden, num_hidden),
			act_func(),

			nn.Linear(num_hidden, num_outputs),

		)



	def forward(self, inputs, is_diffeomorphism=False): #0.0023
		if is_diffeomorphism:
			y1 = self.nn1(inputs)+inputs
			y= self.nn2_1(y1)*y1+y1
		else:
			y = self.nn3(inputs)+inputs
		return y

	def jacobian(self, inputs):
		return get_jacobian(self, inputs, inputs.size(-1))








def get_jacobian(net, x, output_dims, reshape_flag=True):
	"""

	"""
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	x_m.requires_grad_(True)
	y_m = net(x_m)
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J

