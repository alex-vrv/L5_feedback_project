#!/usr/bin/env python
# coding: utf-8

# # L5 to L2/3 feedback model - v1

# ## Description

# As in model v1, the network consists of the following populations:
#
# Excitatory:
# - L2/3 pyramidal neurons
# - L4 pyramidal neurons
# - L5a pyramidal neurons
# - L5b pyramidal neurons
#
# Inhibitory:
# - PV (in L2/3) interneurons
#
# With the following connections:
# - L5a -> L2/3
# - L5b -> PV
# - PV  -> L2/3
# - L4  -> L2/3
#
#
# L5a/b and L4 receive a (exp_rise,exp_decay) (albeit noisy),
# excitatory input current.
#
# As a consequence, the simulation consists of 4 parts*:
# 1. Adjustment of inhibitory weights to achieve E/I balance ('balancing').
# 2. Simulation only with the L5a population turned off.
# 3. Simulation only with the L5b population turned off.
# 4. Simulation with both L5a & L5b on.
#
# **Tried matching Vm experimental values:**
# 1. Random sampling of weights from exponential dist. (except for L4->L2/3)
# 2. made more excitable PV model
# 	- Cm: 100pF (instead of previous 200)
# 	- consequently changed tau_m : now 10ms
# 3. change in pyramidal cells model (L2/3,L4,L5a,L5b)
#     - Cm: 300pF (instead of previous 200)
# 	- consequently changed tau_m : now 30ms
# 4. Changed baseline PV-->L2/3 weights (pre-balancing initialisation)
# 	- NOW: 2 nS
#
#

# ## Code

# In[ ]:


import os
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import external_signals as ext_sign
from scipy.signal import find_peaks,peak_widths,square
import pandas as pd
from scipy.stats import sem
from matplotlib import cm
from cycler import cycler
from itertools import combinations,permutations



import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

colormap = plt.cm.plasma

def cumulative_histogram(x, orientation = 'ascending'):
    x = x.ravel()
    x_sorted = np.sort(x)
    if orientation == 'descending':
        x_sorted = np.reverse(x_sorted)
    percentile = np.arange(len(x))/np.float(len(x))
    return(x_sorted, percentile)

from IPython.display import HTML, display
def set_background(color):
    script = ("var cell = this.closest('.code_cell');"
              "var editor = cell.querySelector('.input_area');"
              "editor.style.background='{}';"
              "this.parentNode.removeChild(this)").format(color)
    display(HTML('<img src onerror="{}">'.format(script)))

#Output measures
def calc_total_spikes(population_label, balancing_time):
    times = spike_monitor[population_label].t / b2.ms
    mask = times >= balancing_time
    total_spikes = np.sum(mask)
    return (total_spikes)


def calc_rate(population_label, balancing_time, total_time):
    total_seconds = (total_time - balancing_time) / 1000
    total_spikes = calc_total_spikes(population_label, balancing_time)
    rate = total_spikes / population_parameters[population_label]['count'] / total_seconds
    return (rate)

def calc_peak_rate(population_label, smoothing_window_width):
    peak_rate = max(rate_monitor[population_label].smooth_rate(window = 'gaussian', width = smoothing_window_width*b2.ms)/b2.Hz)
    return peak_rate

#calculates mean first spike latency to max values of input
def calc_first_spike_latency(population_label,max_I_timepoints):
    spiking_neurons = np.unique(spike_monitor[population_label].i)
    population_latency_values = []
    population_jitter_values = []
    for spiking_neuron in spiking_neurons:
        first_spike_latencies = []
        spike_times = spike_monitor[population_label].t[spike_monitor[population_label].i == spiking_neuron]/b2.second*1000 # in ms            first_spike_latencies = []
        for max_I in max_I_timepoints[max_I_timepoints<max(spike_times)]:
            spike_times_mask = spike_times[(spike_times>max_I) & (spike_times<max_I+100)]
            if len(spike_times_mask)!=0:
                first_spike = spike_times_mask[0]
                first_spike_latencies.append(first_spike - max_I)
        if first_spike_latencies:
            population_latency_values.append(np.mean(first_spike_latencies))
            if len(first_spike_latencies)>1:
                population_jitter_values.append(np.std(first_spike_latencies))

    return np.around(np.mean(population_latency_values),2), np.around(np.mean(population_jitter_values),2)

def plot_first_spike_latency(population_label,max_I_timepoints):
    spiking_neurons = np.unique(spike_monitor[population_label].i)
    population_latency_values = []
    population_jitter_values = []
    for spiking_neuron in spiking_neurons:
        first_spike_latencies = []
        spike_times = spike_monitor[population_label].t[spike_monitor[population_label].i == spiking_neuron]/b2.second*1000 # in ms            first_spike_latencies = []
        for max_I in max_I_timepoints[max_I_timepoints<max(spike_times)]:
            spike_times_mask = spike_times[(spike_times>max_I) & (spike_times<max_I+100)]
            if len(spike_times_mask)!=0:
                first_spike = spike_times_mask[0]
                first_spike_latencies.append(first_spike - max_I)
        if first_spike_latencies:
            population_latency_values.append(np.mean(first_spike_latencies))
            if len(first_spike_latencies)>1:
                population_jitter_values.append(np.std(first_spike_latencies))
    #PLOT
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 5))

    for ax, values, label in zip(axes,[population_latency_values, population_jitter_values], ["First spike latency (ms)","First spike jitter (ms)"]):
        ax.scatter(np.random.uniform(1,1.3,len(values)),values, marker = ".", alpha = 0.5)
        ax.errorbar(1.5, np.mean(values), yerr=np.std(values),marker = "d", c="y", markersize = 10)
        ax.set_ylabel(label)
        ax.tick_params( axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
        ax.set_xlim([0.8,1.7])
    return fig


# ### Saving directory

# In[ ]:

# ##############
# iter=0########
# ##############





for iter in np.arange(5,9,1):




    #folder label indicating simulation details
    #notation: modeltype_stimulustype_activepopulationsorder
    simulation_label = ''.join(('test_discrete_synchronous_L5a_L5b_temporaloffset_10ms_200mswindow_iter',str(iter)))
    if not os.path.exists("simulations/discrete_synchronous/"+simulation_label):
        os.mkdir("simulations/discrete_synchronous/"+simulation_label)

    # locked_offset = 0
    #plot saving flag
    save_plots = True


    # ### Functions for output measures

    # In[ ]:




    # ### Define external input

    # #### General

    # In[ ]:


    b2.start_scope()

    total_neurons = 1000
    balancing_time = 5000
    interval_length = 3000
    only_whisker_time = interval_length
    offset_duration = 10 #in ms
    temporal_window = 200 #overall window in ms

    # test_all = []

    # for i in np.arange(-50,60,10):

    #     positive_diff = i+10
    #     if positive_diff<60:
    #         test_all.append(tuple([i,positive_diff]))
    #     negative_diff = i-10
    #     if negative_diff>-60:
    #         test_all.append(tuple([i,negative_diff]))

    epochs = []
    epochs.append('Only L4 input')
    # offsetL5a = []
    # offsetL5b = []
    for offset in np.arange(int(-temporal_window/2),int(temporal_window/2+offset_duration),int(offset_duration)):
    # for offset_pair in iteration:
        epochs.append('L5a '+str(offset)+' L5b '+str(offset)+' ms')

    # for offset1,offset2 in test_all:
    #     epochs.append('L5a '+str(offset1)+' L5b '+str(offset2)+' ms')
    #     offsetL5a.append(offset1)
    #     offsetL5b.append(offset2)
    # epochs.append('L5a '+str(0)+' L5b '+str(0)+' ms')
    # offsetL5a.append(0)
    # offsetL5b.append(0)


    total_tests = len(epochs)-1
    testing_time = total_tests * interval_length
    total_time = balancing_time + only_whisker_time + testing_time




    # rythmic external input
    duty_cycle_length     = 500
    total_duty_cycles     = int((total_time + duty_cycle_length)/duty_cycle_length)
    duty_cycle_amplitude  = 100
    tau_rise = -5
    tau_decay = -0.5
    step_duration = 200


    # #### L4

    # In[ ]:



    l4_neurons_are_active_input = ext_sign.generate_step_exprisedecay_signal(total_time = total_time,
                                                                             duty_cycle_length = duty_cycle_length,
                                                                             duty_cycle_amplitude=duty_cycle_amplitude,
                                                                             offset=0,
                                                                             tau_rise = tau_rise,
                                                                             tau_decay = tau_decay)


    # #### L5a

    # In[ ]:



    l5a_neurons_are_active_input = ext_sign.generate_step_exprisedecay_signal(total_time = total_time-testing_time,
                                                                             duty_cycle_length = duty_cycle_length,
                                                                             duty_cycle_amplitude=duty_cycle_amplitude,
                                                                             offset=0,
                                                                             tau_rise = tau_rise,
                                                                             tau_decay = tau_decay)

    for counter,offset in enumerate(np.arange(int(-temporal_window/2),int(temporal_window/2+offset_duration),int(offset_duration))):
    # for counter,offset in enumerate(np.arange(int(-temporal_window/2),int(temporal_window/2+offset_duration),int(offset_duration))):
    #     if counter < len(epochs):
        l5a_neurons_are_active_input_temp = ext_sign.generate_step_exprisedecay_signal(total_time = interval_length,
                                                                                             duty_cycle_length = duty_cycle_length,
                                                                                             duty_cycle_amplitude=duty_cycle_amplitude,
                                                                                             offset=offset,
                                                                                             tau_rise = tau_rise,
                                                                                             tau_decay = tau_decay)
    #     else:
    #         l5a_neurons_are_active_input_temp = 0*ext_sign.generate_step_exprisedecay_signal(total_time = interval_length,
    #                                                                                              duty_cycle_length = duty_cycle_length,
    #                                                                                              duty_cycle_amplitude=duty_cycle_amplitude,
    #                                                                                              offset=offset,
    #                                                                                              tau_rise = tau_rise,
    #                                                                                              tau_decay = tau_decay)

        l5a_neurons_are_active_input = np.concatenate((l5a_neurons_are_active_input,l5a_neurons_are_active_input_temp))


    # l5a_neurons_are_active_input = np.concatenate((l5a_neurons_are_active_input,np.zeros(only_whisker_time)))


    # #### L5b

    # In[ ]:




    l5b_neurons_are_active_input = ext_sign.generate_step_exprisedecay_signal(total_time = total_time-testing_time,
                                                                         duty_cycle_length = duty_cycle_length,
                                                                         duty_cycle_amplitude=duty_cycle_amplitude,
                                                                         offset=0,
                                                                         tau_rise = tau_rise,
                                                                         tau_decay = tau_decay)


    for counter,offset in enumerate(np.arange(int(-temporal_window/2),int(temporal_window/2+offset_duration),int(offset_duration))):

    # for counter,offset in enumerate(np.arange(int(-temporal_window/2),int(temporal_window/2+offset_duration),int(offset_duration))):
    #     if counter < len(epochs):
        l5b_neurons_are_active_input_temp = ext_sign.generate_step_exprisedecay_signal(total_time = interval_length,
                                                                                             duty_cycle_length = duty_cycle_length,
                                                                                             duty_cycle_amplitude=duty_cycle_amplitude,
                                                                                             offset=offset,
                                                                                             tau_rise = tau_rise,
                                                                                             tau_decay = tau_decay)
    #     else:
    #         l5b_neurons_are_active_input_temp = 0*ext_sign.generate_step_exprisedecay_signal(total_time = interval_length,
    #                                                                                              duty_cycle_length = duty_cycle_length,
    #                                                                                              duty_cycle_amplitude=duty_cycle_amplitude,
    #                                                                                              offset=offset,
    #                                                                                              tau_rise = tau_rise,
    #                                                                                              tau_decay = tau_decay)

        l5b_neurons_are_active_input = np.concatenate((l5b_neurons_are_active_input,l5b_neurons_are_active_input_temp))

    # l5b_neurons_are_active_input = np.concatenate((l5b_neurons_are_active_input,np.zeros(only_whisker_time)))



    # #### Adding noise & constructing TimedArrays

    # In[ ]:


    # arr_rythmic = np.concatenate((balancing_time_input,only_whisker_input,l5a_neurons_are_active_input,l5b_neurons_are_active_imput))
    I_rythmic_L4 = b2.TimedArray(l4_neurons_are_active_input*b2.pA, dt=1*b2.ms)
    I_rythmic_L5a = b2.TimedArray(l5a_neurons_are_active_input*b2.pA, dt=1*b2.ms)
    I_rythmic_L5b = b2.TimedArray(l5b_neurons_are_active_input*b2.pA, dt=1*b2.ms)

    # # random background noise (time-varying)
    # noise_time_scale = 10.
    # # arr_noise = 50 + 50 * np.random.randn(int(total_time / noise_time_scale), total_neurons)
    # arr_noise = duty_cycle_amplitude + duty_cycle_amplitude * np.random.randn(int(total_time / noise_time_scale), total_neurons)
    # I_random  = b2.TimedArray(arr_noise*b2.pA, dt=noise_time_scale*b2.ms)
    # random background noise (time-varying)
    noise_time_scale = 10

    arr_noise = duty_cycle_amplitude/2 + duty_cycle_amplitude/2 * np.random.randn(int(total_time / noise_time_scale), total_neurons)

    # I_random_L4  = b2.TimedArray(np.ones(total_time)*(duty_cycle_amplitude * np.random.randn(int(total_time / noise_time_scale), total_neurons))*b2.pA, dt=noise_time_scale*b2.ms)
    I_random_L4  = b2.TimedArray(arr_noise*b2.pA, dt=noise_time_scale*b2.ms)
    I_random_L5a = b2.TimedArray(arr_noise*b2.pA, dt=noise_time_scale*b2.ms)
    I_random_L5b = b2.TimedArray(arr_noise*b2.pA, dt=noise_time_scale*b2.ms)


    # ### Inspect external input

    # In[ ]:


    #### General#### General#### General#### L4#### L5a#### L5b### Inspect external inputfig = plt.figure(figsize = (15,5))
    fig = plt.figure(figsize = (15,5))
    plt.plot((I_rythmic_L5a.values*(10**(12)))[0:1000])
    plt.ylabel("Amplitude (pA)")
    stim_f = total_duty_cycles/total_time*1000
    plt.title("Stimulus frequency: %.2f Hz" % stim_f)
    fig.axes[0].text(0.8,-0.1,simulation_label,
            horizontalalignment='left',
            verticalalignment='top',
            transform=fig.axes[0].transAxes)
    max_I_timepoints = find_peaks(I_rythmic_L4.values)[0]

    # if save_plots == True:
    # fig.savefig('simulations/discrete_synchronous/'+simulation_label+'/stimulus.pdf')





    # In[ ]:


    # max_I_timepoints


    # ### Define network parameters

    # In[ ]:



    # Neuron groups
    population_parameters = dict()
    population_parameters['L4']   = {'name' :'L4',   'count':int(total_neurons / 5), 'color':'salmon'}
    population_parameters['L2/3'] = {'name' :'L2/3', 'count':int(total_neurons / 5), 'color':'red'}
    population_parameters['PV']   = {'name' :'PV',   'count':int(total_neurons / 5), 'color':'cornflowerblue'}
    population_parameters['L5a']  = {'name' :'L5a',  'count':int(total_neurons / 5), 'color': colormap.colors[0]}
    population_parameters['L5b']  = {'name' :'L5b',  'count':int(total_neurons / 5), 'color': colormap.colors[-1]}

    # Neuron properties
    E_L   = -60.   *b2.mV       # Resting potential
    E_R   = -80.   *b2.mV       # Inhibitory reversal potential
    V_th  = -50.   *b2.mV       # Spiking threshold
    t_ref =   5.   *b2.ms       # Refractory period
    gl    =  10.   *b2.nsiemens # Leak conductance
    C_m   = 300.   *b2.pfarad   # Membrane capacitance
    C_m_PV   = 100. *b2.pfarad   # Membrane capacitance PV interneuron
    tau_m = C_m/gl              # Membrane time constant
    tau_m_PV = C_m_PV/gl        # Membrane time constant PV interneuron
    gmax  = 100.   *b2.nsiemens # Maximum inhibitory weight
    I_ext =   0.   *b2.pA       # Constant external input current (where applicable)

    base_model = '''
    dv/dt    = -(v - E_L)/tau_m - I/C_m + I_ext/C_m : volt (unless refractory)
    dg_ex/dt = -g_ex/tau_ex_decay                   : siemens
    dg_in/dt = -g_in/tau_in_decay                   : siemens
    I_ex     =  g_ex*v                              : amp
    I_in     =  g_in*(v - E_R)                      : amp
    I        =  I_ex + I_in                         : amp
    dA/dt    =  -A / tau_m                          : siemens
    '''

    base_model_PV = '''
    dv/dt    = -(v - E_L)/tau_m_PV - I/C_m_PV + I_ext/C_m_PV : volt (unless refractory)
    dg_ex/dt = -g_ex/tau_ex_decay                   : siemens
    dg_in/dt = -g_in/tau_in_decay                   : siemens
    I_ex     =  g_ex*v                              : amp
    I_in     =  g_in*(v - E_R)                      : amp
    I        =  I_ex + I_in                         : amp
    dA/dt    =  -A / tau_m_PV                       : siemens
    '''

    base_model_L23 = '''
    dv/dt    = -(v - E_L)/tau_m + I_syn/tau_m - I/C_m + I_ext/C_m : volt (unless refractory)
    dg_ex/dt = -g_ex/tau_ex_decay                   : siemens
    dg_in/dt = -g_in/tau_in_decay                   : siemens
    I_ex     =  g_ex*v                              : amp
    I_in     =  g_in*(v - E_R)                      : amp
    I        =  I_ex + I_in                         : amp
    dA/dt    =  -A / tau_m                          : siemens
    I_syn                                           : volt
    '''

    reset = '''
    v = E_L
    A += 1 * nsiemens
    '''

    externally_driven = base_model + '''
    I_ext    =  I_rythmic(t) + I_random(t, i)       : amp
    '''

    net = b2.Network(b2.collect())

    populations = dict()
    for label in ['L4', 'L5a', 'L5b']:
        populations[label] = b2.NeuronGroup(population_parameters[label]['count'],
                                            model      = base_model +f'''
                                            I_ext    =  I_rythmic_{label}(t) + I_random_{label}(t, i)       : amp
                                            ''',
                                            threshold  = 'v > V_th',
                                            reset      = reset,
                                            refractory = t_ref,
                                            method     = 'euler'
            )
    for label in ['L2/3']:
        populations[label] = b2.NeuronGroup(population_parameters[label]['count'],
                                            model = base_model_L23,
                                            threshold  = 'v > V_th',
                                            reset      = reset,
                                            refractory = t_ref,
                                            method     = 'euler'
        )


    for label in ['PV']:
        populations[label] = b2.NeuronGroup(population_parameters[label]['count'],
                                            model      = base_model_PV,
                                            threshold  = 'v > V_th',
                                            reset      = reset,
                                            refractory = t_ref,
                                            method     = 'euler'
        )
    # randomize initial membrane potentials
    for label, population in populations.items():
        population.v = 'E_L + (V_th - E_L) * rand()'

    net.add(populations)


    # ### Define connectivity

    # In[ ]:




    # Synapse parameters
    # w_exc        =   1.  *b2.nsiemens  # Excitatory weight
    # w_inh        =   1.  *b2.nsiemens  # Inhibitory weight
    tau_ex_decay =   5.  *b2.ms        # Glutamatergic synaptic time constant
    tau_in_decay =   10.  *b2.ms        # GABAergic synaptic time constant
    eta          =   0.1               # Learning rate
    tau_stdp     =  20.  *b2.ms        # STDP time constant
    rho          =  10.  *b2.Hz        # Target excitatory population rate
    beta         =  rho * tau_stdp * 2 # Target rate parameter

    # Connection probabilities for L5 -> L2/3
    p = dict()
    p[('L5a', 'L2/3')] = 10
    # p[('L5b', 'L2/3')] = 10
    # p[('L5a', 'PV')  ] = 10
    p[('L5b', 'PV')  ] = 10
    p[('PV',  'L2/3')] = 10
    p[('L4',  'L2/3')] = 10

    w_mean = dict()
    w_mean[('L5a', 'L2/3')] = 4. * b2.nsiemens
    # w_mean[('L5b', 'L2/3')] = 2. * b2.nsiemens
    # w_mean[('L5a', 'PV')  ] = 2. * b2.nsiemens
    w_mean[('L5b', 'PV')  ] = 4. * b2.nsiemens
    w_mean[('PV',  'L2/3')] = 5. * b2.nsiemens
    w_mean[('L4',  'L2/3')] = 5. * b2.nsiemens

    # rescale
    p_scale = 1 / 100
    p = {k : p_scale * v for k, v in p.items()}

    w_scale = 0.5
    w_mean = {k : w_scale * v for k, v in w_mean.items()}

    connections = dict()
    for (source_label, target_label) in [('L5a', 'L2/3'), ('L5b', 'PV')]:
        source = populations[source_label]
        target = populations[target_label]
        connections[(source_label, target_label)] = b2.Synapses(source, target,
                                                                model='w : siemens',
                                                                on_pre='g_ex_post += w',
                                                                delay = 1. * b2.ms
        )
        connections[(source_label, target_label)].connect(p=p[(source_label, target_label)])
        connections[(source_label, target_label)].w = np.random.exponential(w_mean[(source_label, target_label)],len(connections[(source_label, target_label)]))*b2.siemens

    for (source_label, target_label) in [('L4', 'L2/3')]:
        source = populations[source_label]
        target = populations[target_label]
        connections[(source_label, target_label)] = b2.Synapses(
            source, target,
            model='w : siemens',
            on_pre='g_ex_post += w',
            delay = 1. * b2.ms
            )
    #         '''
    #         w : siemens
    #         delta : siemens
    #         dApre/dt  = -Apre  / tau_stdp : siemens (event-driven)
    #         dApost/dt = -Apost / tau_stdp : siemens (event-driven)
    #         ''',
    #         on_pre='''
    #         Apre += 1.*nsiemens
    #         delta += Apost
    #         g_ex_post += w
    #         ''',
    #         on_post='''
    #         Apost += 1.*nsiemens
    #         delta -= Apre
    #         '''
    #     )
    connections[('L4', 'L2/3')].connect(p=p[('L4', 'L2/3')])
    connections[('L4', 'L2/3')].w = w_mean[(source_label, target_label)]
    # connections[('L4', 'L2/3')].delta = 0. * b2.siemens

    # Vogels-Sprekler synapse to achieve and E/I balance
    connections[('PV', 'L2/3')] = b2.Synapses(
        populations['PV'], populations['L2/3'],
        model= '''
        dz1/dt = - z1 / I_tau_rise         : 1 (clock-driven)
        dz2/dt = (- z2 + z1) / I_tau_decay : 1 (clock-driven)
        I_syn_post = v_I_syn * z2          : volt (summed)
        v_I_syn                            : volt

        w : siemens
        dApre/dt  = -Apre  / tau_stdp : siemens (event-driven)
        dApost/dt = -Apost / tau_stdp : siemens (event-driven)
        ''' ,
        on_pre='''
        Apre += 1.*nsiemens
        w = clip(w + (Apost - beta*nS) * eta, 0, gmax)
        g_in_post += w
        z1 += 1
        ''',
        on_post='''
        Apost += 1.*nsiemens
        w = clip(w + Apre * eta, 0, gmax)
        ''',
        namespace = dict(
        I_tau_rise = 3. * b2.msecond, # IPSP rise time
        I_tau_decay = 40. * b2.msecond # IPSP decay time
        ),
        delay = 0.8 * b2.msecond,
        method = 'euler'
    )

    connections[('PV', 'L2/3')].connect(p=p[('PV', 'L2/3')])
    connections[('PV', 'L2/3')].w = w_mean[(source_label, target_label)]

    net.add(connections)


    # ### Setup monitors

    # In[ ]:


    ### Setup monitors# Create spike, state and rate monitors
    # spike_monitor = dict()
    # state_monitor = dict()
    rate_monitor  = dict()
    for label in populations:
    #     spike_monitor[label] = b2.SpikeMonitor(populations[label])
        # state_monitor[label] = b2.StateMonitor(populations[label], variables=True, record=True)
        rate_monitor[label]  = b2.PopulationRateMonitor(populations[label])
    # net.add(spike_monitor)
    # net.add(state_monitor)
    net.add(rate_monitor)

    # connection_monitor = b2.StateMonitor(connections[('L4', 'L2/3')],
    #                                      variables='delta',
    #                                      dt=1.*b2.ms,
    #                                      record=True)
    # net.add(connection_monitor)


    # ### Run simulation

    # In[ ]:


    # balance the network using inhibitory plasticity
    net.run(balancing_time * b2.ms, report='stdout', report_period=10*b2.second)

    # turn off inhibitory plasticity
    # connections[('PV', 'L2/3')].namespace['eta'] = 0.
    eta = 0.0

    populations['L5a'].active = False
    populations['L5b'].active = False

    net.run(only_whisker_time * b2.ms, report='stdout', report_period=10*b2.second)

    populations['L5a'].active = True
    populations['L5b'].active = True

    net.run(testing_time * b2.ms, report='stdout', report_period=10*b2.second)



    # # turn off L4 neurons
    # populations['L4'].active = False

    # # turn off all L5 neurons
    # populations['L5a'].active = False
    # populations['L5b'].active = False


    # net.run(l4_neurons_are_active * b2.ms, report='stdout', report_period=10*b2.second)

    # # turn back on L5a neurons
    # # turn off L5b neurons
    # populations['L5a'].active = True

    # net.run(l5a_neurons_are_active * b2.ms, report='stdout', report_period=10*b2.second)

    # # turn off L5a neurons
    # # turn back on L5ab neurons
    # populations['L5a'].active = False
    # populations['L5b'].active = True


    # net.run(l5b_neurons_are_active * b2.ms, report='stdout', report_period=10*b2.second)


    # ### Output measures
    #

    # #### Title builder

    # In[ ]:


    simulation_label_split = simulation_label.split('_')
    model_title = simulation_label_split[0].capitalize()
    if 'temporaloffset' in simulation_label_split:
        if 'asynchronous' in simulation_label_split:
            synchronicity_title = 'asynchronous'
            if 'L5bvaried' in simulation_label_split:
                L5a_label = [i for i in simulation_label_split if 'L5a' in i][-1]
                L5a_title = 'L5a locked ('+L5a_label[3:]+'ms)'
                L5b_title = 'L5b varied'
            if 'L5avaried' in simulation_label_split:
                L5a_title = 'L5a varied'
                L5b_label = [i for i in simulation_label_split if 'L5b' in i][-1]
                L5b_title = 'L5b locked ('+L5b_label[3:]+'ms)'
        if 'synchronous' in simulation_label_split:
            synchronicity_title = 'synchronous'

    general_simulation_title = model_title+' model ('+synchronicity_title+' temporal offset)'
    if 'asynchronous' in simulation_label_split:
        general_simulation_subtitle = L5a_title+' - '+L5b_title
        general_simulation_subtitle = general_simulation_subtitle.rjust(int(0.75*len(general_simulation_title)), ' ')
        general_simulation_title = general_simulation_title+'\n'+general_simulation_subtitle


    # In[ ]:


    # set_background('lightblue')

    # print('Total spikes (per population):\n')

    # for label in ['L4', 'L5a', 'L5b', 'L2/3', 'PV']:
    #     print(f"{label} : {calc_total_spikes(label,balancing_time)} spikes")### Output measures


    # In[ ]:


    # set_background('lightblue')

    # print('Total firing rates (population average):\n')

    # for label in ['L4', 'L5a', 'L5b', 'L2/3', 'PV']:
    #     print(f"{label} : {np.around(calc_rate(label,balancing_time,total_time),2)} Hz")


    # In[ ]:


    # set_background('lightblue')

    # smoothingWindowWidth = 25 # in ms
    # print('Peak spike rate (per population):\n')

    # for label in ['L4', 'L5a', 'L5b', 'L2/3', 'PV']:
    #     print(f"{label} : {np.around(calc_peak_rate(label,smoothing_window_width=25),2)} Hz")


    # In[ ]:


    # set_background('lightblue')

    # print('L2/3 first spike latency (L5A): ', calc_first_spike_latency('L2/3',max_I_timepoints[max_I_timepoints>balancing_time])[0], 'ms')
    # print('L2/3 first spike jitter (all epochs): ', calc_first_spike_latency('L2/3',max_I_timepoints[max_I_timepoints>balancing_time])[1], 'ms')

    # print('L2/3 first spike latency (L5a/L5b co-active): ', calc_first_spike_latency('L2/3',max_I_timepoints[max_I_timepoints>3000])[0], 'ms')
    # print('L2/3 first spike jitter (L5a/L5b co-active): ', calc_first_spike_latency('L2/3',max_I_timepoints[max_I_timepoints>3000])[1], 'ms')

    # fig1 = plot_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>balancing_time, max_I_timepoints<2000))])
    # fig1.suptitle(epochs[0], fontsize=16)
    # fig1.axes[0].set_title('L2/3 first spike latency: '+str(calc_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>balancing_time, max_I_timepoints<2000))])[0])+ ' ms')
    # fig1.axes[1].set_title('L2/3 first spike jitter: '+str(calc_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>balancing_time, max_I_timepoints<2000))])[1])+ ' ms')
    # fig1.axes[-1].text(0.5,-0.1,simulation_label,
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=fig1.axes[-1].transAxes)
    # if save_plots == True:
    #     fig1.savefig('simulations/discrete_synchronous/'+simulation_label+'/'+epochs[0]+'_firstspike.pdf')


    # fig2 = plot_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>2000, max_I_timepoints<3000))])
    # fig2.suptitle(epochs[1], fontsize=16)
    # fig2.axes[0].set_title('L2/3 first spike latency: '+str(calc_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>2000, max_I_timepoints<3000))])[0])+ ' ms')
    # fig2.axes[1].set_title('L2/3 first spike jitter: '+str(calc_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>2000, max_I_timepoints<3000))])[1])+ ' ms')
    # fig2.axes[-1].text(0.5,-0.1,simulation_label,
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=fig2.axes[-1].transAxes)
    # if save_plots == True:
    #     fig2.savefig('simulations/discrete_synchronous/'+simulation_label+'/'+epochs[1]+'_firstspike.pdf')


    # fig3 = plot_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>3000, max_I_timepoints<4000))])
    # fig3.suptitle(epochs[2], fontsize=16)
    # fig3.axes[0].set_title('L2/3 first spike latency: '+str(calc_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>3000, max_I_timepoints<4000))])[0])+ ' ms')
    # fig3.axes[1].set_title('L2/3 first spike jitter: '+str(calc_first_spike_latency('L2/3',max_I_timepoints[np.where(np.logical_and(max_I_timepoints>3000, max_I_timepoints<4000))])[1])+ ' ms')
    # fig3.axes[-1].text(0.5,-0.1,simulation_label,
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=fig3.axes[-1].transAxes)

    # if save_plots == True:
    #     fig3.savefig('simulations/discrete_synchronous/'+simulation_label+'/'+epochs[2]+'_firstspike.pdf')


    # ### Plot

    # #### Plot neuronal spike times

    # In[ ]:


    # # plot neuronal spike times
    # fig, axes = plt.subplots(len(population_parameters), 1, sharex=True, figsize=(12, 10))
    # for ax, label in zip(axes, population_parameters):
    #     ax.plot(spike_monitor[label].t/b2.ms, spike_monitor[label].i, '|',
    #             markersize = 3.,
    #             color      = population_parameters[label]['color'],
    #             alpha      = 1.,
    #             rasterized = True)
    #     ax.set_ylabel(population_parameters[label]['name'])
    #     ax.set_xlim(balancing_time, total_time)

    # fig.axes[-1].text(0,-0.2,simulation_label,
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=ax.transAxes)

    # fig.suptitle(general_simulation_title)

    # if save_plots == True:
    #     fig.savefig('simulations/discrete_synchronous/'+simulation_label+'/raster_all_populations.pdf')


    # #### Plot population firing rates

    # In[ ]:


    fig, axes = plt.subplots(len(population_parameters), 1, sharex=True, sharey=True, figsize=(12, 10))

    for ax, label in zip(axes,population_parameters):
        ax.plot(rate_monitor[label].t/b2.ms,
                rate_monitor[label].smooth_rate(window = 'gaussian', width = 25*b2.ms)/b2.Hz,
                label = population_parameters[label]['name'],
                color = population_parameters[label]['color'])
        ax.set_ylabel('Firing rate [Hz]')
        ax.legend(loc='best')
        ax.set_xlim(balancing_time, total_time)
        for interval_start in np.arange(balancing_time,total_time,interval_length*2):
            ax.fill_between(np.arange(interval_start,interval_start+interval_length,1),0,60,alpha=0.05,facecolor='green')




    fig.tight_layout()
    fig.axes[-1].text(0,-0.2,simulation_label,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    fig.suptitle(general_simulation_title)

    if save_plots == True:
        fig.savefig('simulations/discrete_synchronous/'+simulation_label+'/FR_all_populations.pdf')


    # #### Plot L2/3 single neuron Vm
    #
    # - 3 neurons (random sampling)
    #

    # In[ ]:


    # set_background('lightblue')

    # label = 'L2/3'
    # neuron_no = 3
    # fig, axes = plt.subplots(neuron_no+1, 1, sharex=True, figsize=(12, 10))

    # for counter,ii in zip(np.arange(1,(neuron_no+1)),np.random.choice(spike_monitor[label].i, size=neuron_no)):
    #     times, sender = zip(*[(t, i) for (t, i) in zip(spike_monitor[label].t/b2.ms, spike_monitor[label].i) if i == ii])
    #     axes[0].plot(times, sender, '|',
    #                  markersize = 5.,
    #                  color      = population_parameters[label]['color'],
    #                  alpha      = 1.,
    #                  rasterized = True)
    #     axes[0].set_ylabel('Neuron ID')

    #     axes[counter].plot(state_monitor[label].t/b2.ms,
    #             state_monitor[label].variables['v'].get_value()[:,ii]*1000)
    #     axes[counter].set_xlim(balancing_time, total_time)
    #     axes[counter].fill_between(np.arange(1000,2000,1),axes[counter].get_ylim()[0],axes[counter].get_ylim()[1],alpha=0.05,facecolor='green')
    #     axes[counter].fill_between(np.arange(2000,1000,1),axes[counter].get_ylim()[0],axes[counter].get_ylim()[1],alpha=0.05,facecolor='cyan')
    #     axes[counter].fill_between(np.arange(2000,1000,1),axes[counter].get_ylim()[0],axes[counter].get_ylim()[1],alpha=0.05,facecolor='cyan')
    #     axes[counter].set_xlim(balancing_time, total_time)
    #     axes[counter].set_ylabel('Vm (mV)')

    # fig.axes[-1].set_xlabel('Time (ms)')
    # fig.suptitle('L2/3 Vm')
    # fig.align_ylabels()
    # fig.tight_layout()
    # fig.axes[-1].text(0,-0.1,simulation_label,
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         transform=ax.transAxes)


    # if save_plots == True:
    #     fig.savefig('simulations/discrete_synchronous/'+simulation_label+'/single_L23_Vm.pdf')


    # In[ ]:


    from scipy.signal import find_peaks,peak_widths
    import pandas as pd





    peaks_df = pd.DataFrame(columns=['epoch','population','amplitude','hwhm','peak_time'])

    for label in population_parameters:

        FR_temp = rate_monitor[label].smooth_rate(window = 'gaussian', width = 5*b2.ms)/b2.Hz

        for ii,epoch in enumerate(np.arange(balancing_time,total_time,interval_length)):
            epoch_window = np.arange(epoch*10,epoch*10+interval_length*10)
            epoch_rate_temp = FR_temp[epoch_window]

            if label == 'L2/3':
                peaks_temp = find_peaks(epoch_rate_temp,distance = 20*10, height = 0)

                peak_timepoints_indices = []
                for cycle_start in np.arange(0,interval_length*10,duty_cycle_length*10):
                    for index,peak in enumerate(peaks_temp[0]):
                        if cycle_start< peak < cycle_start+40*10:
                            peak_timepoints_indices.append(index)


                peak_time_temp = peaks_temp[0][peak_timepoints_indices]

                hwhm_temp = peak_widths(epoch_rate_temp,peak_time_temp)[0]
                hwhm_temp = hwhm_temp/10

                peak_time_temp = peak_time_temp +epoch*10
                peak_time_temp = peak_time_temp/10


                amplitudes_temp = peaks_temp[1]['peak_heights'][peak_timepoints_indices]

            else:
                peak_time_temp = find_peaks(epoch_rate_temp,distance = duty_cycle_length/5*10, height = 0)[0]

                hwhm_temp = peak_widths(epoch_rate_temp,peak_time_temp)[0]
                hwhm_temp = hwhm_temp/10

                peak_time_temp = peak_time_temp +epoch*10

                peak_time_temp = peak_time_temp/10

                amplitudes_temp = find_peaks(epoch_rate_temp,distance = duty_cycle_length/5*10, height = 0)[1]['peak_heights']
        #         if len(amplitudes_temp)>1:
        #             amplitudes_temp = amplitudes_temp[0]

        #         amplitudes_temp = amplitudes_temp[np.where(peak_time_temp>balancing_time*10)]

        #         amplitudes[label] = amplitudes_temp

        #         hwhm_temp = peak_widths(epoch_rate_temp,peak_time_temp[np.where(peak_time_temp>balancing_time*10)],0.5)[0]




            new_entry = {'epoch':epochs[ii],
                         'population':label,
                         'amplitude':amplitudes_temp[1:],
                         'hwhm':hwhm_temp[1:],
                         'peak_time':peak_time_temp[1:]
                         }


            peaks_df = peaks_df.append(new_entry,ignore_index=True)








    # In[ ]:


    from itertools import chain

    # return list from series of comma-separated strings
    def chainer(s):
        return list(chain.from_iterable(s))

    # calculate lengths of splits
    lens = peaks_df['peak_time'].map(len)
    # create new dataframe, repeating or chaining as appropriate
    res = pd.DataFrame({'epoch': np.repeat(peaks_df['epoch'], lens),
                        'population': np.repeat(peaks_df['population'], lens),
                        'peak_time': chainer(peaks_df['peak_time']),
                        'amplitude': chainer(peaks_df['amplitude']),
                        'hwhm': chainer(peaks_df['hwhm'])})
    if save_plots==True:
        res.to_csv('simulations/discrete_synchronous/'+simulation_label+'/population_peaks.csv')

    # In[ ]:


    colors = sns.color_palette("tab20", len(epochs))

    populations = ['L2/3', 'L4', 'L5a', 'L5b', 'PV']
    populations_multiindex = np.repeat(populations,len(epochs))

    epochs_multiindex = np.tile(epochs,5)

    multiindex_plot = pd.MultiIndex.from_arrays([populations_multiindex,epochs_multiindex],names=('population', 'epoch') )

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,20))


    for i,measure in enumerate(['amplitude','hwhm']):

        mean_s = res.set_index('epoch').groupby(['population', 'epoch'])[measure].mean()
        mean_s = mean_s.reindex(multiindex_plot)
        sem_s = res.set_index('epoch').groupby(['population', 'epoch'])[measure].sem()
        sem_s = sem_s.reindex(multiindex_plot)

        mean_df = mean_s.to_frame()
        mean_df['sem'] = sem_s
        mean_df_plot = mean_df.reset_index()
        pivoted_mean_df = mean_df_plot.pivot(index='population',columns='epoch',values=measure)
        pivoted_mean_df = pivoted_mean_df.reindex(columns = epochs, fill_value =0)
        pivoted_mean_df.plot(ax = axes[i],
                             kind='bar',
                             color = colors,
                             yerr=mean_df_plot.pivot(index='epoch',columns='population',values='sem').values
                             )
        axes[i].set_title('Population response : ' + measure,fontsize = 30)
        axes[i].xaxis.label.set_size(25)
        axes[i].yaxis.label.set_size(25)
        axes[i].legend(fontsize=10, loc='upper center',ncol=2)
        axes[i].tick_params(axis='both', which='major', labelsize=15)





    axes[0].set_ylabel('spk/s', fontsize=20)
    axes[1].set_ylabel('ms', fontsize=20)
    axes[0].titlesize=30
    axes[0].titlesize=30
    axes[1].labelsize=50
    axes[1].labelsize=50

    fig.axes[-1].text(0,0.5,simulation_label,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)

    fig.suptitle(general_simulation_title,fontsize=35)



    plt.subplots_adjust(hspace=0.35)

    if save_plots == True:
        fig.savefig('simulations/discrete_synchronous/'+simulation_label+'/FR_peakamp_hwhm_all.pdf')



    #Plot mean response curves with SEM per epoch

    test_L4 = rate_monitor['L4']
    epoch_rate_per_cycle_mean_L4 = np.zeros((duty_cycle_length*10,len(epochs)))
    epoch_rate_per_cycle_sem_L4 = np.zeros((duty_cycle_length*10,len(epochs)))


    for ii,epoch in enumerate(np.arange(balancing_time,total_time,interval_length)):
        epoch_window = np.arange(epoch*10,epoch*10+interval_length*10)
        rate_epoch_temp = (test_L4.smooth_rate(window = 'gaussian', width = 5*b2.ms)/b2.Hz)[epoch_window]

        epoch_rate_per_cycle = np.zeros((duty_cycle_length*10,int(interval_length/duty_cycle_length)))
        for i,cycle in enumerate(np.arange(epoch_window[0],epoch_window[-1]-duty_cycle_length*10,duty_cycle_length*10)):
            cycle_window = np.arange(cycle,cycle+duty_cycle_length*10)
            rate_cycle_temp = (test_L4.smooth_rate(window = 'gaussian', width = 5*b2.ms)/b2.Hz)[cycle_window]
            epoch_rate_per_cycle[:,i] = rate_cycle_temp

        sem_temp = sem(epoch_rate_per_cycle,1)
        epoch_rate_per_cycle_mean_temp = np.mean(epoch_rate_per_cycle,1)

        epoch_rate_per_cycle_mean_L4[:,ii] = epoch_rate_per_cycle_mean_temp
        epoch_rate_per_cycle_sem_L4[:,ii] = sem_temp
    ######

    test_r = rate_monitor['L2/3']
    epoch_rate_per_cycle_mean = np.zeros((duty_cycle_length*10,len(epochs)))
    epoch_rate_per_cycle_sem = np.zeros((duty_cycle_length*10,len(epochs)))


    for ii,epoch in enumerate(np.arange(balancing_time,total_time,interval_length)):
        epoch_window = np.arange(epoch*10,epoch*10+interval_length*10)
    #     print(epoch_window)
        rate_epoch_temp = (test_r.smooth_rate(window = 'gaussian', width = 5*b2.ms)/b2.Hz)[epoch_window]

        epoch_rate_per_cycle = np.zeros((duty_cycle_length*10,int(interval_length/duty_cycle_length)))
        for i,cycle in enumerate(np.arange(epoch_window[0],epoch_window[-1]-duty_cycle_length*10,duty_cycle_length*10)):
            cycle_window = np.arange(cycle,cycle+duty_cycle_length*10)
            rate_cycle_temp = (test_r.smooth_rate(window = 'gaussian', width = 5*b2.ms)/b2.Hz)[cycle_window]
            epoch_rate_per_cycle[:,i] = rate_cycle_temp

        sem_temp = sem(epoch_rate_per_cycle,1)
        epoch_rate_per_cycle_mean_temp = np.mean(epoch_rate_per_cycle,1)

        epoch_rate_per_cycle_mean[:,ii] = epoch_rate_per_cycle_mean_temp
        epoch_rate_per_cycle_sem[:,ii] = sem_temp


    no_rows,remaining = divmod(len(epochs),4)
    fig, axes = plt.subplots(no_rows +1 if remaining else no_rows, 4, sharex=True, sharey=True, figsize=(16, 10))

    axes = np.array(axes)

    for counter,ax in enumerate(axes.reshape(-1)):


    #     ax = plt.axes()
    #     ax.set_prop_cycle(custom_cycler)
        if counter <= len(epochs)-1:
            line1, = ax.plot(epoch_rate_per_cycle_mean[:,counter], color=colors[counter])
            line2, = ax.plot(epoch_rate_per_cycle_mean_L4[:,counter], color = 'k', alpha = 0.8)
            ax.legend((line1, line2), ('L2/3 ('+epochs[counter]+')', 'L4'))
            ax.set_xlim(0,2500)
            ax.set_ylim(0,int(res[res.population.eq('L2/3')].amplitude.max())+1)
            ax.set_xticks(ax.get_xticks().tolist()) # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - IT IS A BUG FROM MATPLOTLIB 3.3.1
            ax.set_xticklabels(np.arange(0,300,50).astype(str))
            if counter in [0,4,8]:
                ax.set_ylabel('spks/s')
            if counter in [8,9,10,11]:
                ax.set_xlabel('ms')

            ax.fill_between(np.arange(0,duty_cycle_length*10),
                            epoch_rate_per_cycle_mean[:,counter]-epoch_rate_per_cycle_sem[:,counter],
                            epoch_rate_per_cycle_mean[:,counter]+epoch_rate_per_cycle_sem[:,counter],
                            alpha=0.3, color=colors[counter],linewidth=0)
            ax.fill_between(np.arange(0,duty_cycle_length*10),
                            epoch_rate_per_cycle_mean_L4[:,counter]-epoch_rate_per_cycle_sem_L4[:,counter],
                            epoch_rate_per_cycle_mean_L4[:,counter]+epoch_rate_per_cycle_sem_L4[:,counter],
                            alpha=0.3, color= 'k',linewidth=0)



    fig.suptitle(general_simulation_title, fontsize=16)

    fig.tight_layout()

    if save_plots==True:
        fig.savefig('simulations/discrete_synchronous/'+simulation_label+'/L23_mean_response_curves.pdf')


    plt.close("all")

    print(''.join(('Simulation ',str(iter),' finished.')))
