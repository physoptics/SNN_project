from brian2 import *

class BrainModel:
    def __init__(self, input_size):
        self.duration = 1000 * ms
        self.input_size = input_size

        # Synapse constants
        self.gmax = 0.1
        self.we = 0.1
        self.wi = 0.1

        # Initialize input with zeroes
        self.poisson_input = PoissonGroup(input_size, rates=0 * Hz)

        # Excitatory neuron parameters
        taue = 20 * ms
        v_rest_e = -65 * mV
        Vth_e = -52 * mV

        neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / taue : volt
        I_synE = ge * nS * -v                                   : amp
        I_synI = gi * nS * (-100.*mV - v)                        : amp
        dge/dt = -ge / (1.0*ms)                                  : 1
        dgi/dt = -gi / (2.0*ms)                                  : 1
        '''

        # Inhibitory neuron parameters
        taui = 10 * ms
        v_rest_i = -60 * mV
        Vth_i = -40 * mV

        neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / taui : volt
        I_synE = ge * nS * -v                                   : amp
        I_synI = gi * nS * (-85.*mV - v)                         : amp
        dge/dt = -ge / (1.0*ms)                                  : 1
        dgi/dt = -gi / (2.0*ms)                                  : 1
        '''

        # Neuron groups
        num_of_neurons = 500
        self.neurons_e = NeuronGroup(num_of_neurons, neuron_eqs_e,
                                     threshold='v > Vth_e', reset='v = v_rest_e',
                                     method='euler', namespace=self.__dict__)
        self.neurons_i = NeuronGroup(num_of_neurons, neuron_eqs_i,
                                     threshold='v > Vth_i', reset='v = v_rest_i',
                                     method='euler', namespace=self.__dict__)
        self.output_layer = NeuronGroup(2,
                                        'dv/dt = -v / (10*ms) : 1',
                                        threshold='v > 1', reset='v = 0',
                                        method='exact')

        # STDP synapse: poisson_input -> excitatory neurons
        taupre = 20 * ms
        taupost = taupre
        dApre = 0.01 * self.gmax
        dApost = -dApre * taupre / taupost * 1.05

        self.s1 = Synapses(self.poisson_input, self.neurons_e,
            '''
            w : 1
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            ''',
            on_pre='''
            ge += w
            Apre += dApre
            w = clip(w + Apost, 0, gmax)
            ''',
            on_post='''
            Apost += dApost
            w = clip(w + Apre, 0, gmax)
            ''',
            method='euler'
        )
        self.s1.namespace.update({
            'taupre': taupre, 'taupost': taupost,
            'dApre': dApre, 'dApost': dApost,
            'gmax': self.gmax
        })
        self.s1.connect(i='j')
        self.s1.w = 'rand() * gmax'

        # Excitatory -> Inhibitory (one-to-one)
        self.s2 = Synapses(self.neurons_e, self.neurons_i, on_pre='ge += we')
        self.s2.namespace.update({'we': self.we})
        self.s2.connect(i='j')

        # Inhibitory -> Excitatory (lateral inhibition)
        self.s3 = Synapses(self.neurons_i, self.neurons_e, on_pre='gi += wi')
        self.s3.namespace.update({'wi': self.wi})
        self.s3.connect(condition='i != j')

        # Excitatory -> Output
        self.s4 = Synapses(self.neurons_e, self.output_layer, on_pre='ge += we')
        self.s4.namespace.update({'we': self.we})
        self.s4.connect()

        # Monitors
        self.weight_monitor = StateMonitor(self.s1, 'w', record=[0, 1])
        self.input_spike_monitor = SpikeMonitor(self.poisson_input)
        self.output_spike_monitor = SpikeMonitor(self.output_layer)
        self.output_potential_monitor = StateMonitor(self.output_layer, 'v', record=True)
        self.output_hidden_layer_e = SpikeMonitor(self.neurons_e)
        self.output_hidden_layer_i = SpikeMonitor(self.neurons_i)

    def run(self):
        run(self.duration)

    def set_input_rates(self, rates_vector):
        assert len(rates_vector) == self.input_size
        self.poisson_input.rates = rates_vector * Hz

    def get_monitors(self):
        return {
            "weight_monitor": self.weight_monitor,
            "input_spike_monitor": self.input_spike_monitor,
            "output_spike_monitor": self.output_spike_monitor,
            "output_potential_monitor": self.output_potential_monitor,
            "hidden_e_spike_monitor": self.output_hidden_layer_e,
            "hidden_i_spike_monitor": self.output_hidden_layer_i
        }
