"""PACKAGES"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from scipy import sparse
from numpy import linalg as LA
from numpy.linalg import inv

"""ESSENTIAL FUNCTIONS"""

def mse_loss1(output, target):
	"""Loss function"""
	Nt = output.size()[1]
	batch_size = output.size()[0]
	
	mask = np.zeros((batch_size, Nt, 1), dtype="intp")
	
	#setting mask=1 where expectations are not met
	for tr in range(batch_size):
	
		for t in range(Nt):
			if target[tr,t,:]!=-1.:
				mask[tr,t,:]=1
		"""
		plt.figure()
		plt.plot(target[tr,:,:], label = "Target")
		plt.plot(output.detach().numpy()[tr,:,:], label = "Output")
		plt.plot(mask[tr,:,:], label = "Evaluation period")
		plt.legend()
		plt.show()
		plt.close()"""
									
	mask = torch.from_numpy(mask)
	loss_tensor = (mask*(target - output)).pow(2).mean(dim=-1)
	#print(loss_tensor)
	loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
	#print(loss_by_trial)
	return loss_by_trial.mean()

def mse_loss2(output, target):
	"""Loss function"""
	Nt = output.size()[1]
	batch_size = output.size()[0]
	
	mask = np.zeros((batch_size, Nt, 1), dtype="intp")
	
	#setting mask=1 where expectations are not met
	for tr in range(batch_size):
	
		for t in range(Nt):
			if target[tr,t,:]>0.:
				mask[tr,t,:]=1
			
	mask = torch.from_numpy(mask)
	loss_tensor = (mask*(target - output)).pow(2).mean(dim=-1)
	loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
	return loss_by_trial.mean()

def data1(net, trials, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig):
	"""Generate data"""
	#initialising
	dt = net.dt
	input_size=net.input_size
	output_size=net.output_size
	hidden_size=net.hidden_size
	
	time = np.arange(0, Nt, dt) #time grid
	
	inputt = np.zeros((trials, Nt, input_size))
	targett = np.zeros((trials, Nt, output_size))
	
	O_inp = np.zeros((trials, Nt))
	
	#criteria
	ct2 = np.random.rand(trials)<perc #trials without cue gives boolean list
	
	#Time of bursts setting in and out
	if SO_on == 0.0:
		O_on_dev = np.zeros(trials)
	else:
		O_on_dev = np.random.randint(-SO_on, SO_on, trials) #time grid for burst signal deviations
		
	if SO_off == 0.0:
		O_off_dev = np.zeros(trials)
	else:
		O_off_dev = np.random.randint(-SO_off, SO_off, trials)
	
	#looping through all trials	
	for tr in range(trials):
		
		#case where cue occurs	
		if ct2[tr] == False:
			
			mask_burst = np.zeros(Nt) #signal time
			mask_cost = np.zeros(Nt) #onset of cost evaluation
			mask_nocost = np.zeros(Nt)
			for t in time:
				if ((O_on+O_on_dev[tr])<=t) & ((O_on+O_on_dev[tr]+O_off+O_off_dev[tr]>=t)).all():
					mask_burst[t] = 1
				if ((O_on+O_on_dev[tr])<=t) & (cost_onset>=t).all():
					mask_nocost[t] = 1			
				if cost_onset<=t:
					mask_cost[t] = 1
			
			#Creatung data
			O_inp[tr,mask_burst[:]==1] = sig #input signal amplitude
			targett[tr, mask_nocost[:]==1., 0] = -1. #period without evaluation
			t = thresh*(O_off+O_off_dev[tr]) #interpreted time
			targett [tr, mask_cost[:]==1., 0] = t #target threshold
			
	inputt[:,:,0] = O_inp
	"""
	plt.figure()
	for i,c in {0:"c", 1:"m", 2:"y"}.items():
		plt.title(f"Sample data-set")
		plt.xlabel(f"Time")
		plt.ylabel("Signal magnitude")
		plt.plot(O_inp[i,:], label="Input", color = c)
		plt.plot(targett[i,:,0], label="Target", color = c, linestyle = "dotted")
	plt.legend()
	plt.show()
	plt.close()"""
	
	#dtype = torch.FloatTensor
	dtype = torch.float32
	inputt = torch.from_numpy(inputt).type(dtype)
	targett = torch.from_numpy(targett).type(dtype)

	return inputt, targett

"""CLASSES"""

class Net1(nn.Module):
	def __init__(self, tau, dt, input_size, output_size, hidden_size, g_G, seed, h0):
		super(Net1, self).__init__()
		"""Network initialisation"""
		
		#initialising parameters
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.tau = tau
		self.dt = dt
		self.g_G = g_G
		self.non_linearity = torch.tanh
				
		#initialising weights
		
		self.G = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=False)
		self.I = nn.Parameter(torch.Tensor(hidden_size, input_size), requires_grad=True)
		self.w = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=True)
		self.h0 = nn.Parameter(torch.Tensor(hidden_size), requires_grad=False)

		with torch.no_grad():
			
			self.w.normal_(std=1. / np.sqrt(hidden_size), generator=torch.manual_seed(seed))
			self.G.normal_(std=1. / np.sqrt(hidden_size), generator=torch.manual_seed(1))
			self.I.normal_(generator=torch.manual_seed(seed))
			if h0 == True:
				self.h0.zero_()
			else:
				self.h0.normal_()

	def forward(self, input, return_dynamics=False, return_frdynamics=False):
		"""Forward simulation with input"""
	
		batch_size = input.shape[0] #batch size
		seq_len = input.shape[1] # signal time length
		h = self.h0 #initial neuron signals
		r = self.non_linearity(h)
		#r = h
		output = torch.zeros(batch_size, seq_len, self.output_size)
		if return_dynamics:
			traj = torch.zeros(batch_size, seq_len, self.hidden_size)
		if return_frdynamics:
			frtraj = torch.zeros(batch_size, seq_len, self.hidden_size)
		

		#simulation loop	
		for t in range(seq_len):
			h = h + self.dt/self.tau * (-h + self.g_G*r.matmul(self.G.t()) + input[:, t, :].matmul(self.I.t()))
			r = self.non_linearity(h)
			output[:, t, :] = r.matmul(self.w)
			
			if return_dynamics:
				traj[:, t, :] = h
			if return_frdynamics:
				frtraj[:, t, :] = r
		
		if not return_dynamics and not return_frdynamics:
			return output
		if return_dynamics and not return_frdynamics:
			return output, traj
		if not return_dynamics and return_frdynamics:
			return output, frtraj
		else:
			return output, traj, frtraj
				
	def train(self, n_epochs, batch_size, learning_rate, trials_train, series, saved_model, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig):
		"""Training a network"""
		#defining training data
		train_input, train_target = data1(self, trials_train, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig) 

		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
		num_examples = train_input.size()[0]
		
		loss_history = []
		gradient_history = []
		for epoch in range(n_epochs):
			print(f"______epoch {epoch}______")
			
			for t in range(num_examples // batch_size):
			#for t in range(2):
		
				print("batch ",t)
			
				#choose random sample
				random_batch_idx = random.sample(range(num_examples), batch_size)
				batch = train_input[random_batch_idx]
				target = train_target[random_batch_idx]
				#forward
				optimizer.zero_grad()
				output = self.forward(batch)
					
				loss = mse_loss1(output, target)
					
				if np.isnan(loss.item()):
					loss.detach()
					output.detach()
					print("NaN")
					break
							
				print("loss ", loss.item())
				loss_history.append(loss.item())
				#backward
				loss.backward()
				#record gradients
				x = []
				tot = 0
				for param in [p for p in self.parameters() if p.requires_grad]:
					tot += (param.grad ** 2).sum()
					x.append(np.sqrt(tot))
				gradient_history.append(np.asarray(x))
				#optimize
				optimizer.step()	
				#finalize
				loss.detach()
				output.detach()
					
			#saving entire model
			torch.save(self.state_dict(), saved_model+".pth")
			
			#saving current state
			state = {'epoch': epoch, 'state_dict': self.state_dict(),'optimizer': optimizer.state_dict()}
			torch.save(state, saved_model+"_state.pth")
				
		plt.figure()
		plt.title("Error history")
		plt.xlabel("Training steps")
		plt.ylabel("loss")
		n = len(loss_history)
		plt.plot(range(n), np.asarray(loss_history))
		#plt.show()
		plt.savefig(f"sys_{series}_err.png")
		plt.close()
				
		plt.figure()
		plt.title(f"Gradient history")
		plt.xlabel(f"Training steps")
		plt.ylabel("|d(loss)/d(parameter)|")
		n = len(gradient_history)
		gradient_history = np.array(gradient_history)
		for i in range(len(gradient_history[0])):
			plt.plot(range(n), gradient_history[:,i], label = f"parameter {i+1}")
		plt.legend()
		plt.savefig(f"sys_{series}_param.png")
		#plt.show()
		plt.close()

"""ANALYSIS"""

def evaluate(net, dur_list, trials_test, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig, sig_onset):
	"""Testing performance on different data-examples"""
	colors = ["b", "g", "r", "c", "m", "y", "a", "d"]
	plt.figure()
	for i,O_off in enumerate(dur_list):
		color = colors[i]
		test_input, test_target = data1(net, trials_test, Nt+3000, sig_onset, 0, O_off, 0, thresh, 0, cost_onset, sig)
		with torch.no_grad():
			output = net.forward(test_input)
		y_input = test_input[0,:,0].numpy()
		outputn = output.detach().numpy()
		avg_output = np.mean(outputn[:,:,0],0) #averaging outputs of all trials
		y_target = test_target[0,:,0].numpy()
		plt.title(f"Averaged network output over {trials_test} trial(s)")
		plt.xlabel(f"Time t [{net.dt}ms]")
		plt.ylabel("Signals")
		plt.plot(avg_output, label="Output", color=color)
		plt.plot(y_target, label=f"Target", linestyle="dashed", linewidth=.5, color=color)
		plt.plot(y_input, label = f"Input {O_off}ms", linestyle = "dashed", linewidth=.5, color=color)
		plt.legend()
	#plt.savefig(f"cycles{series}_result{xx}.png")
	plt.show()
	plt.close()

def precision(net, dur_list, trials_test, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig, sig_onset):
	"""Calculating precision"""
	ls_loss = []
	for O_off in dur_list:
		input, target = data1(net, trials_test, Nt, sig_onset, 0, O_off, 0, thresh, 0, cost_onset, sig)
		with torch.no_grad():
			output = net.forward(input)
		loss = mse_loss2(output, target)
		ls_loss.append(loss)
	precision = np.mean(ls_loss)
	print(f"precision = {precision}")
	return precision

def weight_correlation(net):
	"""Correlation between input and output weights"""
	I = net.I.detach().numpy().flatten()
	w = net.w.detach().numpy().flatten()
	I_mag = np.sqrt(I.dot(I))
	w_mag = np.sqrt(w.dot(w))
	correl = (I.dot(w))/(I_mag*w_mag)
	print(f"weight-correlation = {correl}")
	return correl

def signals(net, O_on, Nt, input_len, thresh=.007, cost_onset=4500., sig=2.):
	"""Generating numpx firing rate and output signals"""
	input, target = data1(net, 1, Nt, O_on, 0, input_len, 0, thresh, 0, cost_onset, sig)
	with torch.no_grad():
		output, r_time = net.forward(input, return_frdynamics = True)
		r_time = r_time[0].numpy()
		output = output[0,:,0].detach().numpy()
	return r_time, output

def pca_matrix(net,t1,t2,r, plot=False):
	"""PCA covariance matrix"""
	r_time = r
	cov_matrix = np.empty([net.hidden_size, net.hidden_size]) #covariance matrix
	t1 = int(t1//net.dt) #time index
	t2 = int(t2//net.dt)
	for i in range(net.hidden_size):
		for j in range(net.hidden_size):
			r_i = r_time[t1:t2,i]
			r_j = r_time[t1:t2, j]
			cov = np.sum((r_i - np.mean(r_i)) * (r_j - np.mean(r_j))) / (len(r_i) - 1)
			cov_matrix[i,j] = cov
	e_val, e_vec = LA.eig(cov_matrix)
	if plot == True:
		y = [ele.real for ele in e_val]
		x = [idx for idx, ele in enumerate(e_val)]
		plt.plot()
		plt.scatter(x, y, color = "b")
		plt.title("Eigenvalues of Covariance matrix")
		plt.ylabel('Value')
		plt.xlabel('Index')
		plt.grid()
		plt.show()
		plt.close()	
	return e_val, e_vec

def pca_trials(net, plot=False):
	"""PCA analysis by averaging over time and trials"""
	in1 = 50.
	in2 = 150.
	in3 = 300.
	O_on = 10
	Nt = O_on + 4000
	t1 = O_on + 1500
	t2 = t1 + 2500
	r_time1, output1 = signals(net, O_on, Nt, in1)
	r_time2, output2 = signals(net, O_on, Nt, in2)
	r_time3, output3 = signals(net, O_on, Nt, in3)
	#covariance matrix
	cov_matrix = np.empty([net.hidden_size, net.hidden_size]) #covariance matrix
	t1 = int(t1//net.dt) #time index
	t2 = int(t2//net.dt)
	for i in range(net.hidden_size):
		for j in range(net.hidden_size):
			r_i = r_time1[t1:t2,i]
			r_j = r_time1[t1:t2, j]
			cov1 = np.sum((r_i - np.mean(r_i)) * (r_j - np.mean(r_j))) / (len(r_i))
			r_i = r_time2[t1:t2,i]
			r_j = r_time2[t1:t2, j]
			cov2 = np.sum((r_i - np.mean(r_i)) * (r_j - np.mean(r_j))) / (len(r_i))
			r_i = r_time3[t1:t2,i]
			r_j = r_time3[t1:t2, j]
			cov3 = np.sum((r_i - np.mean(r_i)) * (r_j - np.mean(r_j))) / (len(r_i))
			cov_matrix[i,j] = (cov1+cov2+cov3)*(1/3)
	e_val, e_vec = LA.eig(cov_matrix)
	if plot == True:
		y = [ele.real for ele in e_val]
		x = [idx for idx, ele in enumerate(e_val)]
		plt.plot()
		plt.scatter(x, y, color = "b")
		plt.title("Eigenvalues of Covariance matrix")
		plt.ylabel('Value')
		plt.xlabel('Index')
		plt.grid()
		plt.show()
		plt.close()	
	return e_val, e_vec

def I_partic_ratio(net, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig):
	"""Participation ratio of weight vector I and random network PCA eigenvectors"""
	I = net.I.detach().numpy().flatten()
	with torch.no_grad():
		net.h0 = torch.nn.Parameter(.1 * torch.ones_like(net.h0))
	r_time0, output0 = signals(net, 0, 10000, 0)
	e_val0, e_vec0 = pca_matrix(net, 5000, 10000,r_time0)
	ls_components = []
	for i in range(net.hidden_size):
		c = e_vec0.T[i].dot(I)
		ls_components.append(c)
	components = np.array(ls_components)
	part_ratio = np.square(np.sum(components))/np.sum(np.square(components))
	print(f"I-participation ratio = {part_ratio}")
	return part_ratio

def w_partic_ratio(net, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig):
	"""Participation ration of weight vector w on PCA eigenvectors over threshold period"""
	w = net.w.detach().numpy().flatten()
	e_val, e_vec = pca_trials(net)
	ls_components = []
	for i in range(net.hidden_size):
		c = e_vec.T[i].dot(w)
		ls_components.append(c)
	components = np.array(ls_components)
	part_ratio = np.square(np.sum(components))/np.sum(np.square(components))
	print(f"w-participation ratio = {part_ratio}")
	return part_ratio












	
