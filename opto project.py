import numpy as np
from brian2 import *
import random

# from brian2tools import *

np.random.seed(123)

start_scope()
taum = 20 * ms
taue = 5 * ms
taui = 10 * ms
Vt = -50 * mV
Vr = -60 * mV
El = -49 * mV #parameters
neuron_spacing = 50*umetre

eqs = '''
dv/dt  = (ge+gi-(v-El)+v0)/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
v0:volt
'''
rate = []
P1 = NeuronGroup(400, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,                method='exact')
P1.add_attribute('x')
P1.add_attribute('y')
P1.x = '(i%20)*neuron_spacing'
P1.y = '(i//20)*neuron_spacing'
P2 = NeuronGroup(1600, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,              method='exact')
P2.add_attribute('x')
P2.add_attribute('y')
P2.add_attribute('xcross')
P2.add_attribute('ycross')
P2.x = '(i%40)*neuron_spacing'
P2.y = '(i//40)*neuron_spacing'
P2.xcross = '(i%40)*neuron_spacing//2'
P2.ycross = '(i//40)*neuron_spacing//2'

P1.v = 'Vr+rand()*(Vt - Vr)'
P1.ge = 0 * mV
P1.gi = 0 * mV
P2.v = 'Vr+rand()*(Vt - Vr)'
P2.ge = 0 * mV
P2.gi = 0 * mV
#output neuron
P2[990:1000].v0=0* mV

def distance_matrix(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

we = (60*0.27/10)* mV # excitatory synaptic weight (voltage)
wi = (-20*4.5/10)* mV # inhibitory synaptic weightCe = Synapses(P, P, on_pre='ge += we')
Ci1 = Synapses(P1, P1, on_pre='gi += wi')
Ce1 = Synapses(P1, P1, on_pre='ge += we')
Ci2 = Synapses(P2, P2, on_pre='gi += wi')
Ce2 = Synapses(P2, P2, on_pre='ge += we')

Ce1.connect('i!=j', p='exp(-0.5 * ((P1.x[i] - P1.x[j])**2 + (P1.y[i] - P1.y[j])**2) / neuron_spacing**2)')
Ci2.connect('i>=800', p=0.02)  #connection

class Pnd:  #pair num delay
    def __init__(self, start, end1, end2, ipi, n, tiv):
        self.start = start
        self.end1 = end1
        self.end2 = end2
        self.ipi = ipi * ms
        self.n = n
        self.tiv = tiv


# F1
num_objects = 106
random_starts = np.random.randint(100, 201, num_objects)
random_end1s = np.random.randint(100, 201, num_objects)
random_end2s = np.random.randint(100, 201, num_objects)
random_ipi = np.random.randint(0, 800, num_objects)
random_n = np.random.randint(0, 5, num_objects)

# Create Pnd
objects = [Pnd(start, end1, end2, ipi, n, tiv=0) for start, end1, end2,ipi, n in zip(random_starts, random_end1s, random_end2s,random_ipi, random_n)]


#s_mon = SpikeMonitor(P[0:100])
# mon=StateMonitor(P[1000],variables=True,record=True)
mon1000=SpikeMonitor(P[990:1000])
rate.append(mon1000.num_spikes/second)

ke = Network( P ,Ci, Ce)
ke.store()
output_rates = []

ke.restore()


P.v0[1:80]=600*mV
P.v0[800:820] =600 * mV
run(0.2*ms)
P.v0[1:80]=-600*mV
P.v0[800:820] = -600 * mV
run(0.2*ms)
P.v0=0*mV
for t in range (20):
    s_mon = SpikeMonitor(P[280:380])
    run(10 * ms)
    output_rates.append(s_mon.num_spikes / ( ms))
P.v0[1:80]=600*mV
P.v0[800:820] =600 * mV
run(0.2*ms)
P.v0[1:80]=-600*mV
P.v0[800:820] = -600 * mV
run(0.2*ms)
P.v0=0*mV
run(9.6*ms)
output_rates.append(s_mon.num_spikes / (20 * ms))
for t in range (30):
    s_mon = SpikeMonitor(P[280:380])
    run(10 * ms)
    output_rates.append(s_mon.num_spikes / (10 * ms))



t_values = range(len(output_rates))
plot(t_values, output_rates)
xlabel(r'$\tau$ (ms)')
ylabel('Firing rate (sp/s)');
show()
# #first option 10
# ke.restore()
# P.v0=0*mV
# P[0:40].v0= (Vt - Vr)
# P[800:810].v0= (Vt - Vr)
# run(1000 * ms)
# #second option 01
# ke.restore()
# P.v0=0*mV
# P[50:90].v0=+(Vt - Vr)
# P[810:820].v0=+(Vt - Vr)
# run(1000 * ms)
# #third option 00
# ke.restore()
# P.v0=0*mV
# run(1000 * ms)
# #fourth option 11ke.restore()
# P.v0=rand()*(Vt - Vr)
# P.v0=(Vt - Vr)
# run(1000 * ms)

# rate.append(mon1000.num_spikes/second)
#
# # subplot(131)# # plot(mon.t/ms, mon.v[4000], label='I')
# # plot(mon.t/ms, mon.v[0], label='I')
# subplot(121)
# plot(mon1000.t/ms, mon1000.i, ',k')
# subplot(122)
# plot(s_mon.t/ms, s_mon.i, ',k')
# xlabel('Time (ms)')
# ylabel('Neuron index')
#show()
# print(rate)
