import numpy as np
import scipy.signal as signal
from scipy.stats import beta
from scipy.signal import decimate


def generate_step_signal(step_duration,total_time,duty_cycle_length,duty_cycle_amplitude):
    total_duty_cycles     = int((total_time + duty_cycle_length)/duty_cycle_length)
    all_time_points = np.arange(0, total_time, 1.)
    step_signal = np.zeros(len(all_time_points)) #preallocate the output array
    for step_timepoint in np.arange(duty_cycle_length,total_time,duty_cycle_length):
        step_window = np.arange(step_timepoint,step_timepoint+step_duration)
        step_signal[step_window] = duty_cycle_amplitude
    return step_signal
                
def generate_step_expdecay_signal(step_duration,tau_decay,total_time,duty_cycle_length,duty_cycle_amplitude):
    total_duty_cycles     = int((total_time + duty_cycle_length)/duty_cycle_length)
    all_time_points = np.arange(0, total_time, 1.)
    step_expdecay_signal = np.zeros(len(all_time_points)) #preallocate the output array
    for step_timepoint in np.arange(duty_cycle_length,total_time,duty_cycle_length):
        step_window = np.arange(step_timepoint,step_timepoint+step_duration)
        step_expdecay_signal[step_window] = duty_cycle_amplitude*np.exp(-tau_decay*np.arange(0,len(step_window)))
    return step_expdecay_signal

# def generate_step_exprisedecay_signal(step_duration,tau_decay,total_time,duty_cycle_length,duty_cycle_amplitude):
#     total_duty_cycles     = int((total_time + duty_cycle_length)/duty_cycle_length)
#     all_time_points = np.arange(0, total_time, 1.)
#     step_exprisedecay_signal = np.zeros(len(all_time_points)) #preallocate the output array
#     for step_timepoint in np.arange(duty_cycle_length,total_time,duty_cycle_length):
#         step_window = np.arange(step_timepoint,step_timepoint+step_duration)
#         step_exprisedecay_signal[step_window] = duty_cycle_amplitude*signal.exponential(len(step_window),tau = tau_decay)
#     return step_exprisedecay_signal

def generate_sinusoidal_signal(total_time,duty_cycle_length,duty_cycle_amplitude):
    total_duty_cycles     = int((total_time + duty_cycle_length)/duty_cycle_length)
    all_time_points = np.arange(0, total_time, 1.)
    sinusoidal_signal = duty_cycle_amplitude + duty_cycle_amplitude  * np.sin(2* np.pi *all_time_points / duty_cycle_length)
    return sinusoidal_signal
                
def generate_damped_sinusoidal_signal(decay_constant,total_time,duty_cycle_length,duty_cycle_amplitude):
    total_duty_cycles     = int((total_time + duty_cycle_length)/duty_cycle_length)
    all_time_points = np.arange(0, total_time, 1.)
    damped_sinusoidal_signal = duty_cycle_amplitude + duty_cycle_amplitude  * np.exp(-decay_constant*all_time_points/duty_cycle_length) * np.sin(2* np.pi *allTimePoints / duty_cycle_length)
    return damped_sinusoidal_signal

def generate_beta_waveform(total_time, duty_cycle_length,peak_I, offset = 0, alpha_a = 3, beta_b = 5, peak_input = 10):
    ''' Adapted from "Recurrent interactions in local cortical circuits", Peron et al., Nature, 2020
        https://github.com/jwittenbach/ablation-sim
        
        "For each stimulus presentation, the waveform of the current is modelled with a beta distribution 
        with shape parameters α = 3 and β = 5. The beta distribution is defined on the interval [0, 1], 
        giving a distinct beginning and end to the stimulus. For the chosen shape parameters, the beta 
        distribution has a value of 0 at its end points and a peak at (α − 1)/(α + β − 2). To model the 
        fast touch stimulus, the waveform was stretched in time so that the peak occurs 10 ms after the 
        stimulus starting timepoint with a half-width half-maximum of 12.8ms."
        
        INPUTS: 
        - total_time : (ms)
        - duty_cycle_length : interval duration (ms)
        - peak_I : peak stimulus amplitude (pA)
        - offset : temporal offset (ms) (preset to 0)
    '''
    # create stimulus
    peak = peak_input
    a, b = (alpha_a,beta_b)
    peak0 = 1.0*(a-1)/(a+b-2)
    stretch_factor = peak/peak0
    interval = duty_cycle_length

    dt=0.1
    duration = total_time
    # compute stimulus time-course
    step = dt
    beta_duration = int(np.ceil(stretch_factor/step))
    stim_duration = beta_duration
    interval_duration = int(np.ceil(interval/step))
    sim_duration = int(np.ceil(duration/dt))
    stim = np.zeros(sim_duration)

    single_stim = beta.pdf(1.0*np.arange(stim_duration)/(stim_duration-1), a, b)

    i = 0
    while True:
        start = i*interval_duration+offset
        stop = start + stim_duration
        if stop > sim_duration:
            break
        stim[start:stop] += single_stim
        i += 1

    stim = stim/(dt*np.sum(single_stim))
    stim = decimate(stim,10)
    return peak_I*stim/max(stim)   


def generate_step_exprisedecay_signal(total_time, 
                                      duty_cycle_length, 
                                      duty_cycle_amplitude,
                                      stim_duration = 200,
                                      offset = 0,
                                      tau_rise = -1,
                                      tau_decay = -0.7):

    peak_time = 20


    dt = 10/stim_duration
    x = np.arange(0,10,dt)
    y1 = 1 - np.exp(tau_rise*x[:peak_time+1])
    y2 = max(y1)*np.exp(tau_decay*x[:(len(x)-peak_time-1)])
    y = np.concatenate((y1,y2))
    y = y/max(y)*duty_cycle_amplitude

    # plt.plot(x,y)
    # plt.show()

    single_stim = y

    stim = np.zeros(total_time)

    i = 0
    while True:
        start = i*duty_cycle_length
        stop = start + stim_duration
        if stop > total_time:
            break
        stim[start:stop] += single_stim
        i += 1
        
    padding = np.zeros(500)
    stim = np.concatenate((padding,stim,padding))
    stim = stim[len(padding)-offset:len(padding)+total_time-offset]
    return stim
