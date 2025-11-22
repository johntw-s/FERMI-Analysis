import numpy as np
import h5py
import os
import sys
from glob import glob
from cytoolz import concat
import time
from scipy.signal import find_peaks
import argparse

tofROI = (5500, 7500)

sys.path.append('/sdf/home/j/johntw/dev/')
import johntw_utils as jtw

sys.path.append('/sdf/home/j/johntw/FERMI/')
import Constants as c
import RunInfo as RI
import ECalibSupportFunctions as SF
import CovSupportFunctions as CSF

parser = argparse.ArgumentParser(description='FERMI Beamtime Analysis Preprocessing Script')
parser.add_argument('-r', '--run', type=int, required=True, help='Run Number')
ARGS = parser.parse_args()
run = ARGS.run

out_dir = f'/sdf/scratch/lcls/ds/tmo/tmol1043723/scratch/jtw/'

path = (SF.get_run_path(RI.base_path, RI.beamtime, mode='Beamtime')).format
run_info = RI.load_run_info()
info = run_info[run]

if run in RI.exclude:
    print(f'Run {run} was found in the list of excluded runs. Terminating.')
    exit()
else:
    print(f'Loading data from run {run}')

globbed = sorted(concat(glob(path(r)) for r in [run]))
print(f'Total number of files in Run_{run} are {len(globbed)}')

if len(globbed) >= 50:
    n = 25
else:
    n = len(globbed)

fr, to = 5100, 5300

TOF_All, Back_All, spectra_All, padres = [], [], [], []

for i in range(0, n):
    with h5py.File(globbed[i], 'r') as fh:
        bp = fh['Background_Period'][()]
        bunches = fh['bunches'][...]
        where = bunches % bp != 0
        where1 = bunches % bp == 0
        tofs = fh['digitizer/channel3'][...]
        arrs = np.average(tofs[:, fr:to],1)[:, None] - tofs
        TOF_All.append(arrs[where])
        Back_All.append(arrs[where1])
        spectra_All.append(fh['cosp/HorSpectrum'][...])
        padres.append(fh['/photon_diagnostics/Spectrometer/hor_spectrum'][...])
    del bunches, where

TOF_All = np.vstack(TOF_All)
Back_All = np.vstack(Back_All)

avg_TOF = np.mean(TOF_All, axis=0) - np.mean(Back_All, axis=0)

spectra_All = np.vstack(spectra_All)
padres_all = np.vstack(padres)

spectra_avg = np.mean(spectra_All, axis=0) - np.mean(np.mean(spectra_All, axis=0)[40:120])
padres_avg = np.mean(padres_all, axis=0)

peaks_spec, props_spec = find_peaks(spectra_avg, distance = 50, height = 200, width = 0.3, rel_height = 0.5, prominence = 0.1)
left_spec = np.array(props_spec['left_ips'], dtype=int)
right_spec = np.array(props_spec['right_ips'], dtype=int)
bins_spec = [[left_spec[i], right_spec[i]] for i in range(len(left_spec))]

peaks_pad, props_pad = find_peaks(padres_avg, distance = 500, height = 15000, width = 1, rel_height = 0.5, prominence = 0.5)
left_pad = np.array(props_pad['left_ips'], dtype=int)
right_pad = np.array(props_pad['right_ips'], dtype=int)
bins_pad = [[left_pad[i], right_pad[i]] for i in range(len(left_pad))]

pf_out = f'{out_dir}pf_run{run}.h5'
print(f'\nInitializing peak finding output file at {pf_out}')
with h5py.File(pf_out, 'w') as h5fout:
    h5fout.create_dataset('TOF', data=avg_TOF)
    h5fout.create_dataset('spectra', data=spectra_avg)
    h5fout.create_dataset('padres', data=padres_avg)
    h5fout.create_dataset('peaks_spec', data=peaks_spec)
    h5fout.create_dataset('peaks_pad', data=peaks_pad)
    h5fout.create_dataset('left_spec', data=left_spec)
    h5fout.create_dataset('right_spec', data=right_spec)
    h5fout.create_dataset('left_pad', data=left_pad)
    h5fout.create_dataset('right_pad', data=right_pad)
print('Finished making peak finding output file\n')

data = {}

print(f'Creating full data dictionary from all run files')
for i, glob in enumerate(globbed):
    print(f'\tStarting File {glob.split("/")[-1]}')
    data_iter = CSF.process(glob, left_pad, right_pad, left_spec, right_spec)

    if i == 0:
        for key in data_iter.keys():
            if hasattr(data_iter[key], 'shape'):
                data[key] = {}
                data[key][glob.split("/")[-1]] = data_iter[key]
            else:
                data[key] = data_iter[key]

    else:
        for key in data_iter.keys():
            if hasattr(data_iter[key], 'shape'):
                data[key][glob.split("/")[-1]] = data_iter[key]
            else:
                continue

print('\nStarting to combine run files')
for key in data_iter.keys():
    if isinstance(data[key], dict):
        temp = np.array([data[key][key_iter] for key_iter in data[key].keys()])
        if temp[0].ndim == 1:
            data[key] = np.concatenate(temp)
        else:
            data[key] = np.vstack(temp)
    else:
        continue

print('\nCalculating Outer Products')
tof_axis = np.arange(*tofROI)
tof_slice = slice(*tofROI)
DtDs = np.zeros((tof_axis.shape[0], tof_axis.shape[0]))
Ds = np.zeros(tof_axis.shape[0])
ItIs = np.zeros((3,3))
DtIs = np.zeros((tof_axis.shape[0], 3))
Is = np.zeros(3)
n = 0

for i in range(data['tofs'].shape[0]):
    if i % 100 == 0:
        print(f'\tev {i+1} of {data["tofs"].shape[0]} ({(i+1)/(data["tofs"].shape[0])*100:.1f}%)')
    elif i % 10 == 0:
        print(f'\tev {i+1} of {data["tofs"].shape[0]}')

    Fluct_Array = np.array([data['H1'][i], data['H2'][i], data['H3'][i]])

    DtDs += data['tofs'][i][tof_slice, None] * data['tofs'][i][None, tof_slice]
    DtIs += data['tofs'][i][tof_slice, None] * Fluct_Array[None, :]
    ItIs += Fluct_Array[:, None] * Fluct_Array[None, :]
    Ds += data['tofs'][i][tof_slice]
    Is += Fluct_Array
    n += 1

DtD = DtDs/n
DtI = DtIs/n
ItI = ItIs/n
D = Ds/n
I = Is/n

phases = np.array([data['phase1'], data['phase2'], data['phase3'], data['phase4'], data['phase5']])

op_out = f'{out_dir}op_run{run}.h5'
print(f'\nInitializing outer product output file at {op_out}')
with h5py.File(op_out, 'w') as h5fout:
    h5fout.attrs['SLU_lambda'] = data['lambda_s']
    h5fout.create_dataset('tof_axis', data=tof_axis)
    h5fout.create_dataset('phases', data=phases)
    h5fout.create_dataset('DtD', data=DtD)
    h5fout.create_dataset('DtI', data=DtI)
    h5fout.create_dataset('ItI', data=ItI)
    h5fout.create_dataset('D', data=D)
    h5fout.create_dataset('I', data=I)
    h5fout.create_dataset('n', data=n)
print('Finished!')