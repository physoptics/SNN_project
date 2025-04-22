from brian2 import *


output_neurons = 2
simulation_time = 100 * ms  # Time to present each image


# Parameters used in the excitatory equation:
taue = 20 * ms #time constant that describes the behavior of the excitatory neuron. TBD: consider using 100 ms instead, according to fncom-09-00099.pdf it's supposed to increase classification accuracy
v_rest_e = -65 * mV #reset voltage the excitatory neuron is reset to after reaching Vth
Vth_e = -52 * mV #membrane threshold (excitatory) - when the neuron membrane crosses its membrane threshold, the neuron fires and it membrane potential is reset to:
#ge - conductance of excitatory synapses

neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (taue)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

# Parameters used in the inhibitory equation:
taui = 10 * ms #time constant that describes the behavior of the inhibitory neuron
v_rest_i = -60 *mV #reset voltage the inhibitory neuron is reset to after reaching Vth
Vth_i = -40 * mV #membrane threshold (inhibitory) - when the neuron membrane crosses its membrane threshold, the neuron fires and it membrane potential is reset to:
#gi - conductance of inhibitory synapses


neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (taui)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

# STDP parameters:
tc_pre_ee = 20 * ms
tc_post_1_ee = 20 * ms
tc_post_2_ee = 40 * ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

# STDP equations:
eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0
            '''
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'


# excitatory neurons layer:
num_of_neurons_e = 784
neurons_e = NeuronGroup(num_of_neurons_e, neuron_eqs_e, threshold='v>Vth_e', reset='v = v_rest_e',
                      method='euler') #TBD: euler? maybe some other method? check if 'exact' fits better

# inhibitory neurons layer:
num_of_neurons_i = 784
neurons_i = NeuronGroup(num_of_neurons_i, neuron_eqs_i, threshold='v>Vth_i', reset='v = v_rest_i',
                      method='euler') #TBD: euler? maybe some other method? check if 'exact' fits better


input_to_e_neurons = Synapses(poisson_input, neurons_e,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )

e_neurons_to_i_neurons = Synapses(neurons_e, neurons_i, on_pre='v_post += 1*mV')

input_to_e_neurons.connect()
e_neurons_to_i_neurons.connect()
#S.w = 'rand() * gmax'
#mon = StateMonitor(S, 'w', record=[0, 1])
#s_mon = SpikeMonitor(poisson_input)




# Second option for the LIK equations:
# membrane_voltage_excitatory = 'dv/dt = ((Erest - v) + ge*(Eexc - v) + gi*(Einh - v)) / (taue)'
# membrane_voltage_inhibitory = 'dv/dt = ((Erest - v) + ge*(Eexc - v) + gi*(Einh - v)) / (taui)'
#Erest =  * ms #resting membrane potential
#Eexc = * ms #the equilibrium potential of excitatory synapses
#Einh = #the equilibrium potential of inhibitory synapses

'''data_np = data.cpu().numpy()
            plt.figure()
            for i, im in enumerate(data_np):
                plt.subplot(2,2,i+1)
                plt.imshow(np.squeeze(im), cmap='gray')
            plt.show()'''
'''# s2 - STDP parameters:
            tc_pre_ee = 20 * ms
            tc_post_1_ee = 20 * ms
            tc_post_2_ee = 40 * ms
            nu_ee_pre = 0.0001  # learning rate
            nu_ee_post = 0.01  # learning rate
            wmax_ee = 1.0
            exp_ee_pre = 0.2
            exp_ee_post = exp_ee_pre
            STDP_offset = 0.4
'''
'''s2 = Synapses(neurons_e, neurons_i,
            post2before                            : 1.0
                    dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                    dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                    dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0,
                on_pre=pre = 1.; w -= nu_ee_pre * post1,
                on_post=post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1,) #TBD: what STDP rule to use for this connection?
            s2.connect() #TBD - p = 0.02
            s2.w = 'rand() * gmax'
            mon2 = StateMonitor(s2, 'w', record=[0, 1])
            s_mon2 = SpikeMonitor(poisson_input)

            s3 = Synapses(neurons_i, output_layer,
            post2before                            : 1.0
                    dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                    dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                    dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0,
                on_pre=pre = 1.; w -= nu_ee_pre * post1,
                on_post=post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1,) #TBD: what STDP rule to use for this connection?)
            s3.connect()  # TBD - p = 0.02
            s3.w = 'rand() * gmax'
            mon3 = StateMonitor(s2, 'w', record=[0, 1])
            s_mon3 = SpikeMonitor(poisson_input)
            run(100 * second, report='text')'''