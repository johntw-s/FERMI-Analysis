import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit, minimize
from scipy import interpolate
from scipy.signal import find_peaks
from tabulate import tabulate
from glob import glob
from cytoolz import concat 
import Constants as c

def load_tof(path_fmt, run, electron_spectrum_length, mode='standard'):
    '''
    Load in time of flight spectrum array from h5 files (last column is seed_lambda)

    Inputs
        path_fmt (str): string of format type that leads to path of file when called with the run number
        run (int): run number
        electron_spectrum_length (int): number of indices to load in
        mode (str): either "standard" or "background sub"

    Returns
        data (np.array): tof spectra and last column is seed laser wavelength (mode standard)

    '''
    if mode == 'standard':
        globbed = globbed = sorted(concat(glob(path_fmt(r)) for r in [run]))
        data = np.vstack([process(g, electron_spectrum_length) for g in globbed])

        return data
    
    elif mode == 'background sub':
        fr_ele, to_ele = 0, 10000
        fr_BL, to_BL = 0, 4000
        TOF_All, Back_All = [], []

        globbed = globbed = sorted(concat(glob(path_fmt(r)) for r in [run]))
        for ij in range(len(globbed)):
            with h5py.File(globbed[ij], 'r') as f:
                bp = f['Background_Period'][()]
                bunches = f['bunches'][...]
                where = bunches % bp != 0
                where1 = bunches % bp == 0
                tofs = f['digitizer/channel3'][:, fr_ele:to_ele].astype('int64')
                seed_lambda = float(f['photon_source/SeedLaser/Wavelength'][()])

                tofs_corr = np.average(tofs[:, fr_BL:to_BL], 1)[:, None] - tofs
                TOF_All.append(tofs_corr[where])
                Back_All.append(tofs_corr[where1])

        TOF_All = np.vstack(TOF_All)
        Back_All = np.vstack(Back_All)

        Avg_TOF = np.mean(TOF_All, axis=0) - np.mean(Back_All, axis=0)

        return Avg_TOF, seed_lambda
    
    else:
        return ValueError('Mode not recognized')

def stline(x, slope, inter):
    '''
    Plots a linear function that follows slope, intercept with some input independent axis

    Inputs
        x (np.array): input independent array
        slope (float): slope of line
        inter (float): intercept of line

    Returns
        y (np.array): resulting dependent values
    '''
    y = slope*x + inter
    return y

def Transformcoeff(lambda_seed, Harms, Ip, T0, peaks):
    '''
    Calibration from ToF to energy using a stline fit of energy v. (T-T0)**(-2) using the harmonics of the system

    Inputs
        lambda_seed (float): Wavelength of seed laser
        Harms (list): List of harmonics in decreasing order e.g., [10, 9, 8, 7]
        Ip (float): Ionization potential
        T0 (float): Time 0 of spectrometer
        peaks (np.array): Harmonics detected by the peak finder, lower ToF to higher ToF (high to low energy)
    
    Returns
        popt1[0] (float): slope of energy calibration
        popt1[1] (float): intercept of energy calibration
    '''

    seed_energy = 1239.84193/lambda_seed
    energies = [harm*seed_energy - Ip for harm in Harms]
    T2quad = [(10**18)/(pk-T0)**2 for pk in peaks] # in s^{-2}
    popt1, pcov1 = curve_fit(stline, T2quad, energies, p0=[10**(-11),3])
    return popt1[0], popt1[1]

def Transformcoeff1(T0, peaks, energies):
    '''
    Calibration from ToF to energy using a stline fit of energy v. (T-T0)**(-2), using peaks and their known energies

    Inputs
        T0 (float): Time 0 of spectrometer
        peaks (np.array): Peaks of interest detected by the peak finder, lower ToF to higher ToF (high to low energy)
        energies (np.array): Photoelectron energies
    
    Returns
        popt1[0] (float): slope of energy calibration
        popt1[1] (float): intercept of energy calibration
    '''

    T2quad = [(10**18)/(pk-T0)**2 for pk in peaks] # in s^{-2}
    popt1, pcov1 = curve_fit(stline, T2quad, energies, p0=[10**(-11),3])
    return popt1[0], popt1[1]

def Transform(tof, slope, constant):
    '''
    Transformation from ToF to energy spectrum using the parameters from the calibration curve

    Inputs
        tof (np.array): Average ToF of data with multiple peaks (preferably no IR data)
        slope (float): slope of energy calibration
        constant (float): intercept of energy calibration

    Returns
        f3 (np.array): Energy Spectrum
        xnew (np.array): New energy axis
    '''
    fr, to = 5200, 8000
    t0 = max(enumerate(tof[4900:5050]), key=(lambda x: x[1]))[0]+4900
    Energy, E_Spec, radvec = [], [], []

    for ii in range(fr,to):
        E_ev = (10**18)*(slope)/((ii-t0)**2) + constant
        E_S = (10**-27)*((ii-t0)**3)*(tof[ii])/(2*slope)
        Energy.append(E_ev)
        E_Spec.append(round(float(E_S),11))

    f2 = interpolate.interp1d(Energy, E_Spec, kind='cubic')
    xnew = np.linspace(Energy[0], Energy[-1], num=4000, endpoint=True)
    f3 = f2(xnew)

    return f3, xnew

def shapes(filename):
    '''
    Return necessary array shape
    '''

    with h5py.File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]

    return (bunches % bp != 0).sum()

def process(filename, electron_spectrum_length):
    '''
    Processes an h5 file to extract and process the electron data
    '''

    try:
        with h5py.File(filename, 'r') as f:
            bp = f['Background_Period'][()]
            bunches = f['bunches'][...]
            tofs_ele = f['digitizer/channel3'][:, :electron_spectrum_length].astype('float64')
            arrs_ele = np.average(tofs_ele[:, 1500:2550], 1)[:, None] - tofs_ele
            seed_lambda = round(float(f['photon_source/SeedLaser/Wavelength'][()]), 3)
            where = (bunches % bp != 0)
            processed_arrs = arrs_ele[where, :]
            seed_lambda_arr = np.full((processed_arrs.shape[0], 1), seed_lambda)

            return np.concatenate((processed_arrs, seed_lambda_arr), axis=1)
        
    except FileNotFoundError:
        print(f'Error: The file "{filename}" was not found.')
        return None
    
    except KeyError as e:
        print(f'Error: Missing or incorrect key in h5 file: {e}')
        return None
    
    except Exception as e:
        print(f'An unexpected error occured: {e}')
        return None
    
def get_run_path(base_path, beamtime=None, mode='raw', by='name'):
    '''
    Retrieve path to the run given the possible parameters
    '''

    if mode == 'raw':
        return os.path.join(base_path, beamtime, 'Test', 'Run_{:03d}', 'rawdata', '*.h5')
    
    elif mode == 'reduced':
        return os.path.join(base_path, beamtime, 'Test', 'Run_{:03d}', 'work', '*_reduced.h5')
    
    elif mode == 's3s':
        return os.path.join(base_path, beamtime, 'Test', 'Run_{:03d}', 'work', '*_s2s.h5')
    
    elif mode == 'Beamtime':
        return os.path.join(base_path, beamtime, 'Beamtime', 'Run_{:03d}', 'rawdata', '*.h5')
    
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
def plot_peakfound(tof, show=True):
    '''
    Takes the ToF spectrum, and plots it along with the hitfound results

    Inputs
        tof (np.array): Time of flight spectrum values
        show (bool, optional): whether to run plt.show() at the end

    Returns
        peaks (np.array): Array of found peaks
    '''

    peaks, props = find_peaks(tof, distance=2, height=0.1, width=0.1, rel_height=0.2, prominence=1)
    left=np.array(props['left_ips'], dtype=int)
    right=np.array(props['right_ips'], dtype=int)
    shifti = len(peaks)
    bins = [[left[i], right[i]] for i in range(len(peaks))]

    peak_T0, props_T0 = find_peaks(tof, distance=2, height=0.1, width=0.1, rel_height=0.2, prominence=1)
    left_T0 = np.array(props_T0['left_ips'], dtype=int)
    right_T0 = np.array(props_T0['right_ips'], dtype=int)
    shifti_T0 = len(peaks) - 1

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))

    ax1.plot(tof, c='k', linewidth=1)
    for b in bins:
        ax1.axvspan(*b, facecolor='c', alpha=0.5)
    ax1.plot(peaks, tof[peaks], 'v', 'r')
    ax1.set_xlabel('Time (ns)',fontsize=12)
    ax1.set_ylabel('Intensity (arb. U)',fontsize=12)
    ax1.set_xlim([4900,7800])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    ax1.set_title(f'Peak finding')


    ax2.plot(tof, c='k', linewidth=1)
    ax2.plot(peak_T0, tof[peak_T0], 'v', 'r')
    ax2.set_xlabel('Time (ns)',fontsize=12)
    ax2.set_ylabel('Intensity (arb. U)',fontsize=12)
    ax2.set_xlim([4990,5000])
    ax2.set_ylim([-0.01, 5])
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='minor', labelsize=12)
    ax2.set_title(f'T0 = {peak_T0[0]}')
    plt.tight_layout()

    if show:
        plt.show()

    return peaks

def harmonic_energies(harmonics, seed_lambda, molecule, IPdict):
    '''
    Generates a table for the expected photon energies of the harmonics in eV

    Inputs
        harmonics (list): list of harmonics to calculate energies for
        seed_lambda (float): wavelength of the seed laser
        molecule (str): either CO2, Ne, or He
        IPdict (dict): dictionary of ionization potentials

    Returns
        tbl (tabulate): table to be printed
    '''

    seed_energy = 1239.8/seed_lambda

    table_list = []
    headers = ['State/Molecule',*[f'Harmonic {i}' for i in harmonics]]

    if molecule == 'CO2':
        for state in ['X', 'A', 'B', 'C']:
            E = np.array(harmonics)*seed_energy - IPdict[state]
            table_list.append([state, *E])
            
    elif molecule == 'He':
        E = np.array(harmonics)*seed_energy - IPdict['He']
        table_list.append(['He', *E])

    elif molecule == 'Ne':
        E = np.array(harmonics)*seed_energy - IPdict['Ne']
        table_list.append(['Ne', *E])

    else:
        raise ValueError('Unexpected molecule')

    return tabulate(table_list, headers=headers, numalign='center', tablefmt='grid') 

def calib(tof, electron_spectrum_length, peaks, energies_or_harmonics, seed_lambda, analysis_type, IP=None, show=False):
    '''
    Generates a table for the expected photon energies of the harmonics in eV

    Inputs
        tof (np.array): time of flight spectrum
        electron_spectrum_length (int): how far into spectrum you are plotting
        peaks (np.array): peaks of harmonics/known energy
        energies_or_harmonics (list): list of harmonics to plot (descending order), or the known energies of the peaks 
        seed_lambda (float): wavelength of the seed laser
        analysis_type (str): either "harmonics" or "energies"
        IP (float, optional): IP for calibration peaks
        show (bool, optional): whether to run plot show

    Returns
        slope (float): slope of calibration
        intercept (float): intercept of calibration
        energies_ev (np.array): energy axis
        Energy_Spectrum (np.array): energy spectrum
    '''
    T0 = 4994
    seed_energy = 1239.8/seed_lambda
    
    if analysis_type == 'energies':
        slope, constant = Transformcoeff1(T0, peaks, energies_or_harmonics)
        
        energies = energies_or_harmonics

    elif analysis_type == 'harmonics':
        if IP is not None:
            slope, constant = Transformcoeff(seed_lambda, energies_or_harmonics, IP, T0, peaks)
            energies = [harm*seed_energy - IP for harm in energies_or_harmonics]
        else:
            raise ValueError('You need to define an IP for the harmonics analysis')

    print(f'Slope: {slope}')
    print(f'Intercept: {round(constant,2)}')
    print(f'T0: {round(T0,2)}')

    Energy_Spectrum, energies_ev = Transform(tof, slope, constant)[:2]
    TT_axis = np.linspace(peaks[0], peaks[-1], electron_spectrum_length)
    T2quad = [10e18/((t-T0)**2) for t in peaks]
    e_range = np.linspace(np.min(T2quad), np.max(T2quad), 1000)

    plt.figure(figsize=(12,5))

    grid = plt.GridSpec(2,2)

    plt.subplot(grid[0,0])
    plt.plot(tof, color='black', linewidth=1)
    plt.xlabel('Time (ns)')
    plt.ylabel('Intensity (arb. u.)')
    plt.title('Time of Flight Spectrum')
    plt.xlim([4900, 8000])
    for ij in peaks:
        plt.axvline(ij, c='c', alpha=0.3)

    plt.subplot(grid[0,1])
    plt.plot(energies_ev, Energy_Spectrum, color='black')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity (arb. u.)')
    plt.title('Energy Calibrated Spectrum')
    plt.xlim([10, 60])
    for ij in energies:
        plt.axvline(ij, c='c', alpha=0.3)

    plt.subplot(grid[1,:])
    # plt.plot(e_range, SF.stline(e_range, slope, constant), '--r')
    plt.plot(T2quad, energies, 'ko', linewidth=1)
    plt.xlabel('$(T-T_0)^{-2}$ (s$^{-2}$)')
    plt.ylabel('Kinetic Energy (eV)')
    plt.title('Energy Calibration')
    
    plt.tight_layout()
    
    if show:
        plt.show()

    return slope, constant, energies_ev, Energy_Spectrum

def convert_tof_into_eV(tof, slope, intercept, T0, vret, Vref):
    '''
    Takes a time axis and converts it to eV

    Inputs
        tof (np.array): Array of tof spectrum
        slope (float): slope of energy calibration
        intercept (float): intercept of energy calibration
        T0 (int): Time 0

    Returns 
        energy_axis (np.array): Resulting energy
    '''
    tof_corrected = tof - T0
    energy_axis = (1e18 * slope) / (tof_corrected**2) + intercept
    energy_axis = energy_axis+(vret-Vref)
    return energy_axis

def full_spectra_plotting(path, run_info, runs, electron_spectrum_length, slope, intercept, show=False):
    Vref = 0
    vret = 0

    peak = 5946
    T0 = 4994

    plt.figure(figsize=(12,7))

    lines = []
    labels = []

    for i in range(len(runs)):
        tof_temp, seed_lambda_temp = load_tof(path, runs[i], electron_spectrum_length, mode='background sub')

        Normalization_factors = tof_temp[peak]
        time_axis_ns = np.arange(len(tof_temp))
        tof_corrected = time_axis_ns - T0
        tof_corrected = tof_corrected[tof_corrected != 0]

        seed_energy_temp = 1239.84193/seed_lambda_temp
        energy_axis = (1e18 * slope) / (tof_corrected**2) + intercept
        energy_axis = energy_axis + (vret-Vref)
        tof_jacobi_corrected = (10**-27) * (tof_corrected**3) * tof_temp[time_axis_ns != T0] / (2 * slope)
        Normalization_factor_e = tof_jacobi_corrected[peak]

        plt.subplot(2,1,1)
        lines.append(plt.plot(time_axis_ns, tof_temp, linewidth=1)[0])
        labels.append(f'Run {runs[i]}, SLU {["ON" if run_info[runs[i]]["slu"] == 1 else "OFF"][0]}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Intensity (arb. u.)')
        plt.title('Uncalibrated ToF')
        plt.xlim([4900, 8000])
        # plt.legend()

        plt.subplot(2,1,2)
        plt.plot(energy_axis[5000:], tof_jacobi_corrected[5000:], linewidth=1, label=f'Run {runs[i]}, SLU {["ON" if run_info[runs[i]]["slu"] == 1 else "OFF"][0]}')
        plt.xlabel('Photoelectron Kinetic Energy (eV)')
        plt.ylabel('Intensity (arb. u.)')
        plt.title('Calibrated PE Spectrum')
        plt.xlim([10, 60])
        # plt.legend()

    if run_info[runs[0]]['sample'] == 'CO2':
        sb_list = (np.array(run_info[runs[0]]['harmonics'][1:]) + np.array(run_info[runs[0]]['harmonics'][:-1]))/2
        
        for i, harm in enumerate(run_info[runs[0]]['harmonics']):
            if i == 0:
                plt.subplot(2,1,1)
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['X'])-(vret-Vref))-intercept))), ls='-', color='r'))
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['A'])-(vret-Vref))-intercept))), ls='-', color='m'))
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['B'])-(vret-Vref))-intercept))), ls='-', color='g'))
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['C'])-(vret-Vref))-intercept))), ls='-', color='b'))
                for j in ['X Harmonic', 'A Harmonic', 'B Harmonic', 'C Harmonic']:
                    labels.append(j)

                plt.subplot(2,1,2)
                plt.axvline(harm*seed_energy_temp-c.IPs['X'], ls='-', color='r')
                plt.axvline(harm*seed_energy_temp-c.IPs['A'], ls='-', color='m')
                plt.axvline(harm*seed_energy_temp-c.IPs['B'], ls='-', color='g')
                plt.axvline(harm*seed_energy_temp-c.IPs['C'], ls='-', color='b')

            else:
                plt.subplot(2,1,1)
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['X'])-(vret-Vref))-intercept))), ls='-', color='r')
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['A'])-(vret-Vref))-intercept))), ls='-', color='m')
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['B'])-(vret-Vref))-intercept))), ls='-', color='g')
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs['C'])-(vret-Vref))-intercept))), ls='-', color='b')

                plt.subplot(2,1,2)
                plt.axvline(harm*seed_energy_temp-c.IPs['X'], ls='-', color='r')
                plt.axvline(harm*seed_energy_temp-c.IPs['A'], ls='-', color='m')
                plt.axvline(harm*seed_energy_temp-c.IPs['B'], ls='-', color='g')
                plt.axvline(harm*seed_energy_temp-c.IPs['C'], ls='-', color='b')

        for i, sb in enumerate(sb_list):
            if i == 0:
                plt.subplot(2,1,1)
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['X'])-(vret-Vref))-intercept))), ls='--', color='r'))
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['A'])-(vret-Vref))-intercept))), ls='--', color='m'))
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['B'])-(vret-Vref))-intercept))), ls='--', color='g'))
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['C'])-(vret-Vref))-intercept))), ls='--', color='b'))
                for j in ['X Sideband', 'A Sideband', 'B Sideband', 'C Sideband']:
                    labels.append(j)

                plt.subplot(2,1,2)
                plt.axvline(sb*seed_energy_temp-c.IPs['X'], ls='--', color='r')
                plt.axvline(sb*seed_energy_temp-c.IPs['A'], ls='--', color='m')
                plt.axvline(sb*seed_energy_temp-c.IPs['B'], ls='--', color='g')
                plt.axvline(sb*seed_energy_temp-c.IPs['C'], ls='--', color='b')

            else:
                plt.subplot(2,1,1)
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['X'])-(vret-Vref))-intercept))), ls='--', color='r')
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['A'])-(vret-Vref))-intercept))), ls='--', color='m')
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['B'])-(vret-Vref))-intercept))), ls='--', color='g')
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs['C'])-(vret-Vref))-intercept))), ls='--', color='b')

                plt.subplot(2,1,2)
                plt.axvline(sb*seed_energy_temp-c.IPs['X'], ls='--', color='r')
                plt.axvline(sb*seed_energy_temp-c.IPs['A'], ls='--', color='m')
                plt.axvline(sb*seed_energy_temp-c.IPs['B'], ls='--', color='g')
                plt.axvline(sb*seed_energy_temp-c.IPs['C'], ls='--', color='b')

        for i, col in zip(['X', 'A', 'B', 'C'],['r','m','g','b']):
            if run_info[runs[1]]['bins'][i] == None:
                continue
                
            plt.subplot(2,1,1)
            for b in run_info[runs[1]]['bins'][i]:
                plt.axvspan(*b, facecolor=col, alpha=0.1)
            plt.subplot(2,1,2)
            for b in convert_tof_into_eV(np.array(run_info[runs[1]]['bins'][i]), slope, intercept, T0, vret, Vref):
                plt.axvspan(*b, facecolor=col, alpha=0.1)

    if run_info[runs[0]]['sample'] in ['He', 'Ne']:
        sb_list = (np.array(run_info[runs[0]]['harmonics'][1:]) + np.array(run_info[runs[0]]['harmonics'][:-1]))/2
        
        for i, harm in enumerate(run_info[runs[0]]['harmonics']):
            if i == 0:
                plt.subplot(2,1,1)
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']])-(vret-Vref))-intercept))), ls='-', color='r'))
                labels.append('Harmonic')

                plt.subplot(2,1,2)
                plt.axvline(harm*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']], ls='-', color='r')

            else:
                plt.subplot(2,1,1)
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((harm*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']])-(vret-Vref))-intercept))), ls='-', color='r')

                plt.subplot(2,1,2)
                plt.axvline(harm*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']], ls='-', color='r')

        for i, sb in enumerate(sb_list):
            if i == 0:
                plt.subplot(2,1,1)
                lines.append(plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']])-(vret-Vref))-intercept))), ls='--', color='r'))
                labels.append('Sideband')

                plt.subplot(2,1,2)
                plt.axvline(sb*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']], ls='--', color='r')

            else:
                plt.subplot(2,1,1)
                plt.axvline(abs(T0+np.sqrt((1e18*slope)/(((sb*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']])-(vret-Vref))-intercept))), ls='--', color='r')

                plt.subplot(2,1,2)
                plt.axvline(sb*seed_energy_temp-c.IPs[run_info[runs[0]]['sample']], ls='--', color='r')

        for i, col in zip(['Other'],['r']):
            if run_info[runs[1]]['bins'][i] == None:
                continue
                
            plt.subplot(2,1,1)
            for b in run_info[runs[1]]['bins'][i]:
                plt.axvspan(*b, facecolor=col, alpha=0.1)
            plt.subplot(2,1,2)
            for b in convert_tof_into_eV(np.array(run_info[runs[1]]['bins'][i]), slope, intercept, T0, vret, Vref):
                plt.axvspan(*b, facecolor=col, alpha=0.1)

    plt.figlegend(lines, labels, loc = 'center right', ncol=1, bbox_to_anchor = (1.15,0.5))
    plt.suptitle(f'{run_info[runs[0]]["sample"]} with Harmonics {run_info[runs[0]]["harmonics"]}')
    plt.tight_layout()
    
    if show:
        plt.show()