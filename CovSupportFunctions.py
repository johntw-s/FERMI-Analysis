import numpy as np
import h5py

def process(filename, left_pad, right_pad, left_spec, right_spec):
    run_no = int(filename.split('_')[-2])

    with h5py.File(filename, 'r') as fh:
        bunches = fh['bunches'][...]
        
        try:
            hor_spectra = fh['/photon_diagnostics/Spectrometer/hor_spectrum'][..., 0:1000].astype('float64')
            ver_spectra = fh['/photon_diagnostics/Spectrometer/vert_spectrum'][..., 80:850].astype('float64')
        except:
            hor_spectra = np.zeros((100, 1000))
            ver_spectra = np.zeros((100, 1000))

        hors = np.fromiter((np.sum(hor_spectra,1)), 'float')
        vers = np.fromiter((np.sum(ver_spectra,1)), 'float')

        try:
            spectra = fh['/cosp/HorSpectrum'][..., :].astype('float64')
        except:
            spectra = np.zeros((100, 1600))

        spectra = spectra - np.average(spectra[:, 40:120], 1)[:, None]

        hars1 = np.fromiter((np.sum(hor_spectra[:, left_pad[0]:right_pad[0]], 1)), 'float')
        hars2 = np.fromiter((np.sum(spectra[..., left_spec[0]:right_spec[0]], 1)), 'float')
        hars3 = np.fromiter((np.sum(spectra[..., left_spec[1]:right_spec[1]], 1)), 'float')

        try:
            intensities = fh['photon_diagnostics/FEL01/I0_monitor/iom_uh_a_pc'][...].astype('float64')
            intensities1 = fh['photon_diagnostics/FEL01/I0_monitor/iom_uh_a'][...].astype('float64')
        except:
            intensities = np.zeros((100,1))
            intensities1 = np.zeros((100,1))

        try:
            phase1 = round(float(fh['photon_source/FEL01/PhaseShifter3/DeltaPhase'][()]), 3)
            phase2 = round(float(fh['photon_source/FEL01/PhaseShifter4/DeltaPhase'][()]), 3)
            phase3 = round(float(fh['photon_source/FEL01/PhaseShifter5/DeltaPhase'][()]), 3)
            phase4 = round(float(fh['photon_source/FEL01/PhaseShifter6/DeltaPhase'][()]), 3)
            phase5 = round(float(fh['photon_source/FEL01/PhaseShifter7/DeltaPhase'][()]), 3)
            lambda_s = round(float(fh['photon_source/SeedLaser/Wavelength'][()]), 3)
            harm_ref = fh['photon_source/FEL01/harmonic_number'][()]
        except:
            phase1, phase2, phase3, phase4, phase5 = 0, 0, 0, 0, 0
            lambda_s = 0
            harm_ref = 0

        try:
            tofs = fh['digitizer/channel3'][:, :].astype('int64')
        except:
            tofs = np.zeros((100, 60000))

        try:
            IR_energies = fh['user_laser/energy_meter/Energy2'][...].astype('float64')
        except:
            IR_energies = np.zeros(100)

        arrs = np.average(tofs[:, 5056:5300], 1)[:, None] - tofs
        # fmt = 'peak{}'.format

        return {
            'bunch': bunches,
            'hor': hors,
            'ver': vers,
            'H1': hars1,
            'H2': hars2,
            'H3': hars3,
            'Run': run_no,
            'intensity_pc': intensities,
            'intensity_uj': intensities1,
            'phase1': phase1,
            'phase2': phase2,
            'phase3': phase3,
            'phase4': phase4,
            'phase5': phase5,
            'lambda_s': lambda_s,
            'harm_ref': harm_ref,
            'IR_energy': IR_energies,
            'tofs': tofs,
            'arrs': arrs
        }