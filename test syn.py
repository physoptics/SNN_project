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
v0: volt
x: metre
y: metre
'''

rate = []

P1 = NeuronGroup(400, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms, method='exact')
P1.x = '(i%20)*neuron_spacing'
P1.y = '(i//20)*neuron_spacing'

P2 = NeuronGroup(1600, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms, method='exact')
P2.x = '(i%40)*neuron_spacing'
P2.y = '(i//40)*neuron_spacing'
P2.add_attribute('xcross')
P2.add_attribute('ycross')
P2.xcross = '(i%40)*neuron_spacing//2'
P2.ycross = '(i//40)*neuron_spacing//2'

P1.v = 'Vr + rand() * (Vt - Vr)'
P1.ge = 0 * mV
P1.gi = 0 * mV
P2.v = 'Vr + rand() * (Vt - Vr)'
P2.ge = 0 * mV
P2.gi = 0 * mV

# Output neuron
P2[990:1000].v0 = 0 * mV

we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight

Ci1 = Synapses(P1, P1, on_pre='gi += wi')
Ce1 = Synapses(P1, P1, on_pre='ge += we')
Ci2 = Synapses(P2, P2, on_pre='gi += wi')
Ce2 = Synapses(P2, P2, on_pre='ge += we')

Ce1.connect('i != j', p='exp(-0.5 * ((x_pre - x_post)**2 + (y_pre - y_post)**2) / neuron_spacing**2)')
Ci2.connect('i >= 800', p=0.02)

# Additional code to define and run your network...
