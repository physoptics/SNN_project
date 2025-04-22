import numpy as np
from brian2 import *
# from brian2tools import *

np.random.seed(123)

start_scope()
taum = 20 * ms
taue = 5 * ms
taui = 10 * ms
Vt = -50 * mV
Vr = -60 * mV
El = -49 * mV

eqs = '''
dv/dt  = (ge+gi-(v-El)+v0)/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
v0:volt
'''

P = NeuronGroup(1000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,                method='exact')

P.v = 'Vr'
P.ge = 0 * mV
P.gi = 0 * mV

#neuron that doesn't get input as marker


#control
controlp = 50 * ms  # Period of the square wave
controlon = 10 * ms  # Duration of the high state of the square wave
controlv = -60 * mV # Amplitude of the square wave

we = (60*0.27/10)* mV # excitatory synaptic weight (voltage)
wi = (-20*4.5/10)* mV # inhibitory synaptic weightCe = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce = Synapses(P, P, on_pre='ge += we')
Ce.connect('i<800', p=0.02)
Ci.connect('i>=800', p=0.02)
s_mon = SpikeMonitor(P[0])# mon=StateMonitor(P[4000],variables=True,record=True)
mon=StateMonitor(P,variables=True,record=True)
P.v0=0*mV
for i in range (10):
    P.v0[1:80]=600*mV
    P.v0[800:820] =600 * mV
    run(0.2*ms)
    P.v0[1:80]=-600*mV
    P.v0[800:820] = -600 * mV
    run(0.2*ms)
    P.v0=0*mV
    run(100*ms)
P.v0=0*mV
run(1000 * ms)



# subplot(131)# # plot(mon.t/ms, mon.v[4000], label='I')
# plot(mon.t/ms, mon.v[0], label='I')
subplot(121)
# plot(mon.t/ms, mon.v[0])
hist(s_mon.i)
subplot(122)
plot(s_mon.t/ms, s_mon.i, ',k')
# plot(mon.t/ms, mon.v[999])
# hist(s_mon.i[999])
xlabel('Time (ms)')
ylabel('Neuron index')
show()

