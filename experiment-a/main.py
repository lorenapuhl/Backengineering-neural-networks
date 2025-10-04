import utils as m
import matplotlib.pyplot as plt
import numpy as np
import torch
import cmath

""" XXXXXXXXXX PARAMETERS XXXXXXXXXX """
#parameters in ms

#network
tau = 100 #time constant
dt= 1 #signal time step size
input_size = 1
output_size=1
hidden_size=500
g_G = 1.5
seed = 1
h0 = True # if True h0 is initialised to zero

#data
trials_train = 500 #number of trials in training data-set
Nt = 7000 #time steps
O_on = 3000 #signal onset time
SO_on = 1000 #deviation of onset time
O_off = 150 #average duration of signal
SO_off = 100 #deviation of signal duration
thresh =.007 # threshold level increase rate
perc = .1
cost_onset = 4500 #time of cost evaluation
sig = 2. #input signal strength

#training
n_epochs = 50 #training epochs 
batch_size = 32
learning_rate = .001

#saving
series = "A" #name series for saving figures A
saved_model = f"sys_{series}"

#analysis
trials_test = 1 #number of trials in evaluation data-set
dur_list = [50, 150, 300] #input-signal lengths
sig_onset = 3000 #onset-time for input signal

#packing parameters
net_pack = (tau, dt, input_size, output_size, hidden_size, g_G, seed, h0)
data_pack = (Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig)
train_pack = (n_epochs, batch_size, learning_rate, trials_train, series, saved_model)

train = 0
ana = 0 # basic network analysis

""" XXXXXXXXXXX TRAINING XXXXXXXXXX """
if train == 1:
    """Setting different initial conditions for I"""
    for i in range(1,100):
        #parameters
        seed = i
        net_pack = (tau, dt, input_size, output_size, hidden_size, g_G, seed, h0)
        series = f"A{seed}"
        saved_model = f"sys_{series}"
        train_pack = (n_epochs, batch_size, learning_rate, trials_train, series, saved_model)

        #initialising
        net = m.Net1(*net_pack)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        net.train(*train_pack, *data_pack)

""" XXXXXXXXXXX ANALYSIS XXXXXXXXXX """

#loading model network
saved_model = f"thresh_X"
net_pack = (tau, dt, input_size, output_size, hidden_size, g_G, seed, h0)
net = m.Net1(*net_pack)
state = torch.load(f"{saved_model}_state.pth")
net.load_state_dict(state['state_dict'])

I = net.I.detach().numpy().flatten()
I_mag = np.sqrt(I.dot(I))
print(I_mag)


#performance
dur_list = [50, 150, 300]
sig_onset = 3000
print(f"\nmodel {saved_model}")
m.evaluate(net, dur_list, trials_test, *data_pack, sig_onset)
m.precision(net, dur_list, trials_test, *data_pack, sig_onset)

ls_data = []
for seed in range(1,21):
    #loading
    saved_model = f"sys_A{seed}"
    net_pack = (tau, dt, input_size, output_size, hidden_size, g_G, seed, h0)
    net = m.Net1(*net_pack)
    state = torch.load(f"{saved_model}_state.pth")
    net.load_state_dict(state['state_dict'])

    #analysis
    print(f"\nmodel {saved_model}")
    #m.evaluate(net, dur_list, trials_test, *data_pack, sig_onset)
    prec = m.precision(net, dur_list, trials_test, *data_pack, sig_onset)
    weight_corr = m.weight_correlation(net)
    I_part = m.I_partic_ratio(net, *data_pack)
    w_part = m.w_partic_ratio(net, *data_pack)
    data = [seed, prec, weight_corr, I_part, w_part]
    ls_data.append(data)

data = np.array(ls_data)
x = data[:,1] #precision
y2 = data[:,2] # weight-correlation
y3 = data[:,3] #I participation ratio
y4 = data[:,4] #w participation ratio
plt.plot()
plt.title("Perfomance analysis")
plt.scatter(x, y2, color = "b", label="weight correlation")
plt.scatter(x, y3, color = "g", label="I-part.ratio")
plt.scatter(x, y4, color = "r", label="w-part.ratio")
plt.ylabel("Quantity of analysis")
plt.xlabel("Performance")
plt.grid()
plt.legend()
plt.savefig("sys_A_performance.png")
plt.show()
plt.close()	


