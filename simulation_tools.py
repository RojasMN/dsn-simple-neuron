import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import gridspec # type: ignore

"""
Examples of the use of this functions can be found in the 'preliminary.ipynb' file

"""

#############################################################################################################################################
###################################################### CURRENT INJECTION GENERATORS #########################################################
#############################################################################################################################################

def generate_ramp_current(time_sim, start_time, duration, start_current, end_current, dt = 0.1):
    
    """
    Generates a ramp current that linearly increases from 'start_current' to 'end_current' 
    over the specified time interval ('start_time' to 'start_time + duration'). 
    Time is in ms and currents are in pA. Returns the resulting current array.

    """
    
    time = np.arange(0, time_sim, dt)
    I = np.zeros_like(time)
    
    # We compute the index of 'start_time' and 'start_time + duration'
    start_idx = int(start_time / dt)
    end_idx = int((start_time + duration) / dt)
    
    # Generate the ramp from 'start_current' to 'end_current'
    I[start_idx:end_idx] = np.linspace(start_current, end_current, end_idx - start_idx)
    
    return I, time


def generate_square_pulse_current(time_sim, start_time, duration, start_current, pulse_current, dt = 0.1):
    
    """
    Generates a square-pulse DC current with a specified amplitude ('pulse_current') 
    applied over a defined time interval ('start_time' to 'start_time + duration'), 
    superimposed on a baseline current ('start_current'). Time is in ms and currents 
    are in pA. Returns the resulting current array.
    
    """
    
    time = np.arange(0, time_sim, dt)
    I = np.ones_like(time) * start_current 
    
    # We compute the index of 'start_time' and 'start_time + duration'
    start_idx = int(start_time / dt)
    end_idx = int((start_time + duration) / dt)
    
    # Set the current to 'pulse_current' during the pulse period
    I[start_idx:end_idx] = pulse_current
    
    return I, time


def generate_zap_current(time_sim, dc_current, start_time, duration, start_freq, end_freq, amplitude, dt = 0.1):
     
    """
    Generates a zap current waveform with a sinusoidal shape whose frequency increases 
    linearly from 'start_freq' to 'end_freq' over the specified 'duration', centered 
    around a DC offset ('dc_current'). The function computes the phase based on 
    the frequency sweep and adds it to the current array within the given time interval. 
    The resulting waveform is returned.

    """
    
    time = np.arange(0, time_sim, dt)
    I = np.ones_like(time) * dc_current 
    end_time = start_time + duration
    
    # First we convert ms to seconds for computing the zap frequency
    t_sec = time / 1000.0
    start_sec = start_time / 1000.0 
    end_sec = end_time / 1000.0
    duration_sec = end_sec - start_sec
    
    # Then, we generate the zap only within the given interval
    zap_mask = (time >= start_time) & (time <= end_time)
    t_zap = t_sec[zap_mask] - start_sec # Relative time in seconds
    
    # When the frequency changes linearly, the change in phase is quadratic
    k = (end_freq - start_freq) / duration_sec
    phase = 2 * np.pi * (start_freq * t_zap + 0.5 * k * t_zap**2)
    
    # Finally we add the zap to the current array
    I[zap_mask] += np.sin(phase) * amplitude
    
    return I, time


def generate_noisy_current(time_sim, amplitude = 5, tau = 5, rate_hz = 5, dt = 0.1):
    
    """
    Generates a noisy current injection. The amplitude value represents the standard deviation of 
    the signal (in pA). The variable tau corresponds to the time constant of the Ornstein-Uhlenbeck process.
    
    """
    
    time = np.arange(0, time_sim, dt)
    n = len(time)
    
    I_noise = np.zeros(n)
    
    # Ornstein-Uhlenbeck process
    for t in range(1, n):
        I_noise[t] = (I_noise[t-1] * np.exp(-dt/tau) + amplitude * np.sqrt(1 - np.exp(-2*dt/tau)) * np.random.normal())
        
    return I_noise, time


def concatenate_currents(currents, dt = 0.1):
    
    """
    Concatenates a list of current waveforms into a single continuous waveform.
    Generates a corresponding time array that spans the total duration of all waveforms,
    ensuring proper alignment without overlap. Each segment's time and current arrays 
    are appended and then concatenated into one final time and current array, which is returned.
    Useful for combining multiple stimulation protocols into a single extended waveform.

    """
    
    # Empty list to store all concatenated currents 
    concatenated_current = []
    concatenated_time = []
    initial_time = 0
    
    # We loop through all currents and concatenate them
    for current in currents:
        current_length = len(current)
        time_array = np.arange(initial_time, initial_time + current_length * dt, dt)
        
        # Append times and currents to the list
        concatenated_time.append(time_array)
        concatenated_current.append(current)
        
        # Finally, we update the time for the next segment
        initial_time += current_length * dt
        
    # Concatenate the currents and time arrays
    concatenated_time = np.concatenate(concatenated_time)
    concatenated_current = np.concatenate(concatenated_current)  
    
    return concatenated_current, concatenated_time


def plot_current_protocol(I, time):
    
    """
    Plots a given current or stimulation protocol
    
    """
    
    plt.figure(figsize=(6, 1))
    plt.plot(time, I)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.grid(False)
    plt.show()

#############################################################################################################################################
############################################### SIMPLE NEURON MODEL (IZHIKEVICH MODEL) ######################################################
#############################################################################################################################################
    
def izhikevich_neuron_sim(C, vr, vt, k, a, b, c, d, vpeak, T, I_inj, dt = 0.1):
    
    """
    Simulates Izhikevich neuron dynamics using Euler integration. Takes neuron parameters (C, vr, vt, k, a, b, c, d, vpeak), 
    simulation duration (T), time step (dt), and input current (I_inj). Updates membrane potential (voltage) and recovery 
    variable (recovery) at each step, with spike reset when voltage ≥ vpeak (reset to c, recovery += d). Returns time array, 
    voltage trace, recovery trace, and spike times. Raises ValueError if I_inj length doesn't match T/dt.
    
    """
    
    # Initialize variables
    n = int(T / dt)                     # Number of steps
    time = np.arange(0, T, dt)          # Time vector
    voltage = np.ones(n) * vr
    recovery = np.zeros(n)
    
    # Check if the stimulation protocol has the same length than the simulation 
    if len(I_inj) != n:
        raise ValueError(f"Current array length ({len(I_inj)}) doesn't match simulation steps ({n})")
    
    I = I_inj
    
    spike_times = []
    
    # Forward Euler integration
    for i in range(n - 1):
        voltage[i+1] = voltage[i] + dt * (k * (voltage[i] - vr) * (voltage[i] - vt) - recovery[i] + I[i]) / C     # Update membrane potential
        recovery[i+1] = recovery[i] + dt * a * (b * (voltage[i] - vr) - recovery[i])                              # Update recovery variable
        
        # Spike detection and reset
        if voltage[i+1] >= vpeak:
            voltage[i] = vpeak
            voltage[i+1] = c
            recovery[i+1] += d
            
            # We save the spike times in a list for further analysis
            spike_times.append(i*dt) 
            
    return time, voltage, recovery, spike_times


def plot_simulation_voltage(time, voltage, protocol, voltage_height_ratio = 3):
 
    # Create figure with gridspec for height ratios
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[voltage_height_ratio, 1])
    
    # Voltage trace (top plot - taller)
    ax0 = plt.subplot(gs[0])
    ax0.plot(time, voltage, color='black')
    ax0.set_ylabel("Membrane Potential (mV)", fontsize = 10)
    ax0.grid(False)
    
    # Remove x-axis labels from top plot
    ax0.set_xticklabels([])
    
    # Current protocol (bottom plot - shorter)
    ax1 = plt.subplot(gs[1])
    ax1.plot(time, protocol, color='orange')
    ax1.set_xlabel("Time (ms)", fontsize = 10)
    ax1.set_ylabel("Current (pA)", fontsize = 10)
    ax1.grid(False)
    
    # Align y-labels
    ax0.yaxis.set_label_coords(-0.08, 0.5)
    ax1.yaxis.set_label_coords(-0.08, 0.5)
    
    plt.tight_layout()
    plt.show()
    
    
def plot_simulation_recovery(time, recovery, protocol, voltage_height_ratio = 3):
 
    # Create figure with gridspec for height ratios
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[voltage_height_ratio, 1])
    
    # Voltage trace (top plot - taller)
    ax0 = plt.subplot(gs[0])
    ax0.plot(time, recovery, color='black')
    ax0.set_ylabel("Recovery Variable", fontsize = 10)
    ax0.grid(False)
    
    # Remove x-axis labels from top plot
    ax0.set_xticklabels([])
    
    # Current protocol (bottom plot - shorter)
    ax1 = plt.subplot(gs[1])
    ax1.plot(time, protocol, color='orange')
    ax1.set_xlabel("Time (ms)", fontsize = 10)
    ax1.set_ylabel("Current (pA)", fontsize = 10)
    ax1.grid(False)
    
    # Align y-labels
    ax0.yaxis.set_label_coords(-0.08, 0.5)
    ax1.yaxis.set_label_coords(-0.08, 0.5)
    
    plt.tight_layout()
    plt.show()
    
#############################################################################################################################################
##################################################### SIMPLE F-I EXPERIMENT + PLOTTING ######################################################
#############################################################################################################################################
    
def simple_frequency_current_protocol(current_steps, neuron_params, time_sim, dt = 0.1):
    
    """
    Measures firing rates of an Izhikevich neuron across DC current steps. For each current in current_steps, 
    injects a constant current lasting time_sim ms, runs the simulation (using izhikevich_neuron_sim), counts spikes, 
    and returns firing rates (Hz) as a list. The neuron_params dictionary should contain all required Izhikevich model 
    parameters (C, vr, vt, k, a, b, c, d, vpeak). Time step (dt) defaults to 0.1 ms.
    
    """
    
    firing_rates = []
    
    for current in current_steps:
        
        # First, we generate a pulse with a given amplitude present in current_steps
        
        """
        Note that we dont need to use the 'generate_square_pulse_current' since 
        the current needed to perform the protocol is a DC current that lasts 
        the entire simulation
        
        """
    
        time = np.arange(0, time_sim, dt)
        I_inj = np.ones_like(time) * current
        
        # Simulation
        time, voltage, recovery, spike_times = izhikevich_neuron_sim(I_inj = I_inj, **neuron_params)
        
        # Calculate the firing frequency
        rate = len(spike_times) / (time_sim / 1000)  # Number of spikes in 500 ms
        firing_rates.append(rate)
    
    return firing_rates


def plot_fi_curve(current_steps, firing_rates):
    
    plt.figure(figsize=(6, 3))
    
    # Plot F-I curve
    plt.plot(current_steps, firing_rates, 'o', markersize = 2)
    plt.xlabel('Current (pA)', fontsize = 10)
    plt.ylabel('Firing Rate (Hz)', fontsize = 10)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.show()
    
#############################################################################################################################################
######################################################### SIMPLE MODEL PHASE DIAGRAM ########################################################
#############################################################################################################################################

def plot_phase_diagram(C, vr, vt, k, a, b, I, v_trace = None, u_trace = None):
    
    """
    Plots the phase diagram with optional simulation trace (as a dashed line).
    No downsampling is applied, so the full trace is plotted.
    
    """
    
    # Grid of (v, u) values
    v = np.linspace(-100, 50, 200)
    u = np.linspace(-100, 100, 200)
    V, U = np.meshgrid(v, u)
    
    # We compute the derivatives (dv/dt and du/dt) for all (v, u)
    dv = (k * (V - vr) * (V - vt) - U + I) / C
    du = a * (b * (V - vr) - U)
    
    # Then we normalize the vectors for better visualization
    magnitude = np.sqrt(dv**2 + du**2)
    dv_norm = dv / magnitude
    du_norm = du / magnitude
    
    # Plot the vector field 
    plt.figure(figsize = (8, 5))
    strm = plt.streamplot(V, U, dv, du, color = magnitude, cmap = 'viridis', density = 1.5, linewidth = 0.4, arrowsize = 0.7)
    plt.colorbar(strm.lines, label = 'Magnitude')
    plt.xlabel('Membrane Potential (mV)')
    plt.ylabel('Recovery Variable')
    plt.xlim((-100, 50))
    plt.ylim((-100, 100))
    plt.grid(True)
    
    # We also add the v- and u-nullclines 
    v_nullcline = k * (v - vr) * (v - vt) + I  
    u_nullcline = b * (v - vr)                 
    plt.plot(v, v_nullcline, 'red', label = 'v nullcline', alpha = 0.5)
    plt.plot(v, u_nullcline, 'blue', label = 'u nullcline', alpha = 0.5)
    
    # Overlay simulation trace (dashed line)
    if v_trace is not None and u_trace is not None:
        plt.plot(v_trace, u_trace, 'k--', lw = 2, alpha = 0.7, label = 'Simulation')  # Dashed line
        plt.scatter(v_trace[0], u_trace[0], color = 'green', marker = 'o', label = 'Start', s = 20)
        plt.scatter(v_trace[-1], u_trace[-1], color = 'red', marker = 'o', label = 'End', s = 20)
    
    plt.legend(bbox_to_anchor = (1.3, 1), loc = 'upper left', borderaxespad = 0.0)
    
    plt.tight_layout()
    plt.show()