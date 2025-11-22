import numpy as np
import pandas as pd
import ast

base_path = '/sdf/data/lcls/ds/tmo/tmol1043723/results/johntw'
beamtime = '20244097'

exclude = [274, 284, 398, 528, 556, 666, 691, 719, 720, 740, 761, 793, 855, 966, 1015, 1148, 1175, 1221, 1244, 1313, 1358] #REMOVE 274 LATER!!!

run_summary = pd.read_csv('/sdf/home/j/johntw/FERMI/Run_Summary.csv', header=1)

def conv_bins(bin_str):
    try:
        return ast.literal_eval(bin_str)
    except:
        return None

def load_run_info():
    run_info = {}

    for i in range(run_summary.shape[0]):
        nums_split = run_summary['Run Number(s)'][i].split('-')
        for j in np.arange(int(nums_split[0]), int(nums_split[-1])+1):
            run_info[j] = {}

            run_info[j]['sample'] = run_summary['Sample'][i]

            if run_summary['Phase Shifter'][i] == 'x':
                run_info[j]['ps'] = None
            else:
                run_info[j]['ps'] = int(run_summary['Phase Shifter'][i])

            run_info[j]['harmonics'] = [int(k) for k in run_summary['Harmonics'][i].split(',')]

            if run_summary['Ref'][i] == 'x':
                run_info[j]['ref'] = None
            else:
                run_info[j]['ref'] = int(run_summary['Ref'][i])

            if run_summary['SLU'][i] == 'ON':
                run_info[j]['slu'] = 1
            elif run_summary['SLU'][i] == 'OFF':
                run_info[j]['slu'] = 0

            run_info[j]['bins'] = {
                'X': conv_bins(run_summary['X State'][i]),
                'A': conv_bins(run_summary['A State'][i]),
                'B': conv_bins(run_summary['B State'][i]),
                'C': conv_bins(run_summary['C State'][i]),
                'Other': conv_bins(run_summary['Other'][i])
            }
    
    return run_info