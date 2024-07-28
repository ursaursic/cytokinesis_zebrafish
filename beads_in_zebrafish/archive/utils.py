'''
Ursa Ursic, last updated: 12.3.2024

Utility file for all the bead motion analysis. 
'''
import pandas as pd
import numpy as np
import os
import cv2
from scipy.optimize import curve_fit, Bounds
import colorcet as cc

import bokeh.io
import bokeh.plotting
import bokeh.models
import iqplot

###
# TODO: 
# - - correct the saving of data
# - - add the option to clear the folders
# - - mkdir if dir does not exist
# - - save as png (as well)

#######
# define force calibration parameters
#######
force_calibration_params = {
    500: [5.59472213e+02, -4.17780556e+01,  1.61519352e+01, -8.84746192e+13],
    '500': [5.59472213e+02, -4.17780556e+01,  1.61519352e+01, -8.84746192e+13],
    1000: [9.00468885e+02, -5.10553973e+01,  1.96828094e+01,  1.05746977e+14],
    '1000': [9.00468885e+02, -5.10553973e+01,  1.96828094e+01,  1.05746977e+14],
    2000: [6.47755226e+02, -5.16372481e+01, 1.87872313e+01, 4.49671461e+08],
    '2000': [6.47755226e+02, -5.16372481e+01, 1.87872313e+01, 4.49671461e+08],
    'RANGE': [9.00468885e+02, -5.10553973e+01,  1.96828094e+01,  1.05746977e+14]
}

def find_tip(tip_file: str, threshold_tip: int = 850, save_to_file=False) -> list:
    img = cv2.imread(tip_file, cv2.IMREAD_UNCHANGED)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    tip_mask = blurred_img < threshold_tip
    tip_mask = np.array(tip_mask.astype(int))
    tip_outline = np.array([[0, 0]])
    tip_end = [0, 0]
    for i in range(200, 800):
        for j in range(1, 300):
            if np.sum(tip_mask[i, j] != tip_mask[i-1:i+1, j-1:j+1])==2:
                tip_outline = np.concatenate([tip_outline, [[j, i]]], axis=0)
                if j > tip_end[0]:
                    tip_end = [j, i]

    if tip_outline.shape[0] > 20: 
        if save_to_file:  
            img = cv2.imread(tip_file, cv2.IMREAD_UNCHANGED)
            filename = os.path.basename(tip_file).split('.')[0]
            p = bokeh.plotting.figure(width=400, height=400, title=f"Yay, we found a tip!!! \n file: {filename}")
            p.title.text_font_size = '16pt'
            p.x_range.range_padding = p.y_range.range_padding = 0
            p.image(image=[img], x=0, y=0, dw=1024, dh=1024)
            p.circle(x=tip_outline[:, 0], y=tip_outline[:, 1], color='red')
            p.star(x=tip_end[0], y=tip_end[1], color='cyan', size=15)
            if save_to_file=='ipynb':
                bokeh.io.show(p)
            else:
                bokeh.io.output_file(filename=f"{save_to_file}", title=f'{filename}')
                bokeh.io.save(p)
        return tip_outline, tip_end

    else:
        print("No tip found... try another threshold.")
        return False, False
    

def add_distance_from_tip(df: pd.DataFrame, tip_point: list[int, int]) -> None:
    '''
    Calculates distance from the tip end point (tip_point) to bead and adds this to the dataframe df.
    '''
    df['DISTANCE [um]'] = np.sqrt((df['POSITION_X']-tip_point[0])**2+(df['POSITION_Y']-tip_point[1])**2)


def add_force(df: pd.DataFrame, calibration: int) -> None:
    '''
    Calculates force at the distance of the bead, depending on the force calibration parameters (defined above).
    '''
    a1, k1, a2, k2 = force_calibration_params[calibration]
    df['FORCE [pN]'] = a1*np.exp(df['DISTANCE [um]']/k1)+a2*np.exp(df['DISTANCE [um]']/k2)


def add_flow_slope(df: pd.DataFrame, first_pulse: int, t_on: int, t_off: int):
    '''
    This function checks out the bead tracks. It takes into account the last 2/3 of the OFF phase before the new period starts. It calculates the slopes where possible and adds them into the data frame. 
    
    ---
    df:             dataframe with bead tracks
    first_pulse:    frame number at which the first pulse starts
    t_on:           length of the ON phase (in number of frames)
    t_off:          length of the OFF phase (in number of frames)
    ---

    Returns none, because the point is to add a column into the dataframe. 
    '''

    df['CORRECTION_k'] = np.nan
    df['CORRECTION_k_ERR'] = np.nan
    df['CORRECTION_N'] = np.nan
    df['CORRECTION_N_ERR'] = np.nan

    periods = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on+t_off)] for j in range(-1, 20)])

    for idx in df['TRACK_ID'].unique():
        track = df[df["TRACK_ID"] == idx]

        for period in periods:
            interval = np.array([period[-1] - t_off*2//3 + i for i in range(0, t_off*2//3)])
            xdata = track[track["FRAME"].isin(interval)]["FRAME"].values
            ydata = track[track["FRAME"].isin(interval)]["DISTANCE [um]"].values
            if len(xdata) >= 5:
                f = lambda x, *p: p[0]*x + p[1]
                popt, pcov = curve_fit(f, xdata, ydata, p0=[-1, 300], nan_policy='raise')
                if (pcov[1][1]/popt[1] < 0.3) & (pcov[0][0]/popt[0] < 0.01):
                    df.loc[(df["TRACK_ID"]==idx) & (df["FRAME"].isin(period)), 'CORRECTION_k'] = popt[0]
                    df.loc[(df["TRACK_ID"]==idx) & (df["FRAME"].isin(period)), 'CORRECTION_k_ERR'] = pcov[0][0]
                    df.loc[(df["TRACK_ID"]==idx) & (df["FRAME"].isin(period)), 'CORRECTION_N'] = popt[1]
                    df.loc[(df["TRACK_ID"]==idx) & (df["FRAME"].isin(period)), 'CORRECTION_N_ERR'] = pcov[1][1]


def calculate_displacement(df: pd.DataFrame, first_pulse: int, t_on: int, t_off: int, substract_background: bool=False):
    '''This function substracts the signal from the background (so that we get increasing displacement after each first point of the new pulse). It creates a new column in the df dataframe with the displacement values after each pulse. 

    ---
    df:             dataframe with bead tracks
    first_pulse:    frame number at which the first pulse starts
    t_on:           length of the ON phase (in number of frames)
    t_off:          length of the OFF phase (in number of frames)
    substract_background: calculate displacement considering the background flows. If true, both displacement and corrected displacement will be calculated.
    ---

    Returns none, because the point is to add a column into the dataframe. 
    '''
    global substracted_background
    substracted_background = substract_background

    periods = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on+t_off)] for j in range(-1, 20)])

    # Calculate displacement without background correction
    df['DISPLACEMENT [um]'] = np.nan
    for idx in df['TRACK_ID'].unique():
        track = df[df["TRACK_ID"] == idx]
        for period in periods:
            data = df.loc[(df['TRACK_ID']==idx) & (df["FRAME"].isin(period)), 'DISTANCE [um]'].values
            if len(data)>0:
                displacement = data[0] - data
                df.loc[(df['TRACK_ID']==idx) & (df["FRAME"].isin(period)), 'DISPLACEMENT [um]'] = displacement

    add_flow_slope(df, first_pulse, t_on, t_off)

    # Calculate dispolacement with background correction
    if substract_background:
        df['CORRECTED DISPLACEMENT [um]'] = np.nan

        # background is calculated as accelerated movement
        background_func = lambda t, t_1, x_1, k_1, k_2: x_1 + k_1*(t-t_1) + (k_2-k_1)/(2*(t_on+t_off))*(t-t_1)**2

        for idx in df['TRACK_ID'].unique():
            track = df[df["TRACK_ID"] == idx]

            for (period_1, period_2) in zip(periods[:-1], periods[1:]):
                popt_1 = track.loc[track["FRAME"]==period_1[0],  ["CORRECTION_k", "CORRECTION_N"]].values
                popt_2 = track.loc[track["FRAME"]==period_2[0],  ["CORRECTION_k", "CORRECTION_N"]].values
                
                if (popt_1.shape != (0,2)) & (popt_2.shape != (0,2)):
                    if not (np.isnan(popt_1).any() or np.isnan(popt_2).any()):
                        k_1, N_1 = popt_1[0]
                        k_2, _ = popt_2[0]
                        x_1 = k_1*period_2[0] + N_1
                        time = track[track["FRAME"].isin(period_2)]["FRAME"].values
                        data = track[track["FRAME"].isin(period_2)]["DISTANCE [um]"].values
                        if len(data) >= 5:
                            corrected_data = background_func(time, time[0], x_1, k_1, k_2) - data 
                            corrected_data -= corrected_data[0] 
                            df.loc[(df['TRACK_ID']==idx) & (df["FRAME"].isin(period_2)), 'CORRECTED DISPLACEMENT [um]'] = corrected_data


def calculate_viscosity(df: pd.DataFrame, magnet_pulses: np.ndarray) -> None:
    '''
    Try not to use this function. 

    Calculate effective viscosity for each pulse. 

    ---
    df:             dataframe with bead tracks
    magnet_pulses:  2D array of pulse times for each pulse
    substracted_background: calculate viscosity from corrected displacement (with background substraction)
    ---
    '''
    for (track, g) in df.groupby('TRACK_ID'):
        for magnet_pulse in magnet_pulses:
                time = g.loc[g['FRAME'].isin(magnet_pulse), 'POSITION_T']
                force = g.loc[g['FRAME'].isin(magnet_pulse), 'FORCE [pN]']
                if substracted_background:
                    displacement = g.loc[g['FRAME'].isin(magnet_pulse), 'CORRECTED DISPLACEMENT [um]']
                else: 
                    displacement = g.loc[g['FRAME'].isin(magnet_pulse), 'DISPLACEMENT [um]']
                displacement_force_ratio = displacement/force

                if len(time)<5 or (~displacement_force_ratio.notna()).any():
                    continue
                
                # fit a linear function to the "magnet on" period and extract viscosity from it
                f = lambda x, *p: p[0]*x + p[1]
                popt, pcov = curve_fit(f, time, displacement_force_ratio, p0=[0.1, 10], nan_policy='raise')
                eff_viscosity = 1000/(popt[0]*6*np.pi*1.4) #mPas
                df.loc[(df['FRAME'].isin(magnet_pulse)) & (df['TRACK_ID']==track), 'EFF_VISCOSITY'] = eff_viscosity


def jeffreys_model_rising(t, k, eta_1, eta_2) -> None:
    f = 1/k * (1 - np.exp(-k*t/eta_1)) + t/eta_2
    return f


def jeffreys_model_relaxing(t, a, tau_r) -> None:
    f = (1-a)*np.exp(-t/tau_r) + a
    return f


def fit_jeffreys_model(data_x: np.ndarray, data_y: np.ndarray, phase: str) -> np.ndarray:
    '''
    fit Jeffrey's rising and relaxing phases to data. Jefferey's model is composed of a sping with an elastic constant k, parralely bound to a dashpod with viscosity eta_1. This component is also sequentially bound to another dashpod with viscosity eta_2. The model is fitted with a "rising pahse", where we assume a constant force at each time step and a "relaxing phase" where the force is 0, but the system is relaxing after the force was turned off. 
          |--- dashpod (eta_1)--|
    |-----|                     |---- dashpod (eta 2) -----> F
          |--- spring (k) ------|
    '''
    if phase not in ['rising', 'relaxing']:
        print('Phase not valid. Options: \'rising\' or \'relaxing\'.')

    data_x_fit = np.linspace(data_x[0], data_x[-1], 100)
    
    try:
        if phase == 'rising':
            bounds = Bounds([0, 0.05, 0.05], [np.inf, np.inf, np.inf]) # k, eta_1, eta_2
            popt, pcov = curve_fit(jeffreys_model_rising, data_x, data_y, bounds=bounds, p0=[100, 1000, 10000])
            data_y_fit = jeffreys_model_rising(data_x_fit, *popt)
        elif phase == 'relaxing':
            bounds = Bounds([0, 0], [np.inf, np.inf]) # a, tau_r
            popt, pcov = curve_fit(jeffreys_model_relaxing, data_x, data_y, bounds=bounds, p0=[0.1, 10])
            data_y_fit = jeffreys_model_relaxing(data_x_fit, *popt)

        
        return data_x_fit, data_y_fit, popt, pcov
    except:
        print("The fit was not sucessful.")
        return [(False,),  (False,),  (False,),  (False,)]


def add_phase(df: pd.DataFrame, phases: np.array) -> None:
    '''
    Add information about the phase into the data frame.
    '''
    for (i, phase) in enumerate(['m_phase_1', 'i_phase_1', 'm_phase_2', 'i_phase_2']):
        if (phases[i*2] != 'na') & (phases[i*2+1] != 'na'):
            df.loc[df['FRAME'].isin(range(int(phases[i*2]), int(phases[i*2+1]))), 'PHASE'] = phase
    print('added phases')


def get_results_mgtw(filename, df, magnet_pulses, df_all_results) -> pd.DataFrame:
    '''
    This is a function used for the data from folder "magnetic_tweezers".
    Generate a new data frame with the calculated results for one data set. Save the new data frame into .csv at save_to_filepath. 
    '''
    for track_idx in df['TRACK_ID'].unique(): 
        for pulse in magnet_pulses:
            pulse_start = pulse[0]
            df_pulse = df[(df['TRACK_ID']==track_idx)&(df['FRAME'].isin(pulse))]
            viscosity = df_pulse['EFF_VISCOSITY'].values
            phase = df_pulse['PHASE'].values
            if len(viscosity) > 0:
                new_result_row = {
                    'FILENAME' : filename, 
                    'TRACK_IDX' : track_idx,
                    'PULSE_START_FRAME' : pulse_start,
                    'MAGNET_STATUS' : 1,
                    'PHASE' : phase[0],
                    }
                df_new_row = pd.DataFrame([new_result_row], columns=df_all_results.columns) 
                df_all_results = pd.concat([df_all_results, df_new_row], ignore_index=True)

    df_all_results = df_all_results[df_all_results['VISCOSITY'].notna()&(df_all_results['VISCOSITY']>0)&(df_all_results['VISCOSITY']<20000)]
    return df_all_results


def get_results_mgtw_time_prec(filename, df, magnet_pulses, df_all_results) -> pd.DataFrame:
    '''
    This is a function used for the data from folder "magnetic_tweezers_time_prec".
    Generate a new data frame with the calculated results for one data set. Save the new data frame into .csv at save_to_filepath. 
    '''
    for track_idx in df['TRACK_ID'].unique(): 
        for pulse in magnet_pulses:
            pulse_start = pulse[0]
            df_pulse = df[(df['TRACK_ID']==track_idx)&(df['FRAME'].isin(pulse))]
            # TODO: viscosity params
            viscosity_params = [0]
            # TODO: define MT state
            if len(viscosity_params) > 0:
                new_result_row = {
                    'FILENAME' : filename, 
                    'TRACK_IDX' : track_idx,
                    'PULSE_START_FRAME' : pulse_start,
                    'MAGNET_STATUS' : 1,
                    'MT_STATUS': 'to be defined',
                    'VISCOSITY' : 'to be defined'
                    }
                df_new_row = pd.DataFrame([new_result_row], columns=df_all_results.columns) 
                df_all_results = pd.concat([df_all_results, df_new_row], ignore_index=True)

    return df_all_results


def plot_trajectories(filename: str, df: pd.DataFrame, first_pulse: int, t_on: int, t_off: int, save_to_filepath: str, show_background_fit=False) -> None:
    p = bokeh.plotting.figure(
    frame_width = 400,
    frame_height = 300,
    x_axis_label='Frame',
    y_axis_label='Distance from tip (um)',
    title=filename
    )
    source_on = bokeh.models.ColumnDataSource(df[df["MAGNET_STATUS"]==1])
    source_off = bokeh.models.ColumnDataSource(df[df["MAGNET_STATUS"]==0])
    p.circle(source=source_on, x='FRAME', y='DISTANCE [um]', alpha=0.5, color='green', legend_label='Magnet ON')
    p.circle(source=source_off, x='FRAME', y='DISTANCE [um]', alpha=0.5, legend_label='Magnet OFF')

    if show_background_fit:
        for idx in df['TRACK_ID'].unique():
            track = df[df["TRACK_ID"] == idx]
            periods = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on+t_off)] for j in range(-1, 20)])

            for period in periods:
                interval = np.array([period[-1] - t_off*2//3 + i for i in range(0, t_off*2//3)])

                xdata = track[track["FRAME"].isin(interval)]["FRAME"].values
                ydata = track[track["FRAME"].isin(interval)]["DISTANCE [um]"].values
                popt = track.loc[track["FRAME"] == interval[0],  ["CORRECTION_k", "CORRECTION_N"]].values
                if popt.size != 0:
                    f = lambda x, k, N: k*x + N  
                    p.circle(x=xdata, y=ydata, alpha=0.3, size=2, color='red', legend_label='Data for drift fit')
                    x_fit = np.linspace(min(xdata), max(xdata), 30)
                    y_fit = f(x_fit, *popt[0])
                    p.line(x=x_fit, y=y_fit, alpha=0.3, line_width=2, color='black', legend_label='Drift fit')

    p.legend.click_policy = 'hide'

    if save_to_filepath=='ipynb':
        bokeh.io.show(p)
    else:
        bokeh.io.output_file(save_to_filepath, title=f'{filename}_trajectories')
        bokeh.io.save(p)


def plot_displacement(filename: str, df: pd.DataFrame, dt: int, save_to_filepath: str) -> None:
    p = bokeh.plotting.figure(
            frame_width = 400,
            frame_height = 300,
            x_axis_label='Time (s)',
            y_axis_label='Displacement (um)',
            title=filename,
            )
    colors = cc.b_glasbey_category10

    for color, (track, g) in zip(colors, df.groupby('TRACK_ID')):
        if substracted_background:
            g = g.dropna(subset=['CORRECTED DISPLACEMENT [um]'])
            displacement = g['CORRECTED DISPLACEMENT [um]']
            if save_to_filepath and save_to_filepath != 'ipynb':
                bokeh.io.output_file(save_to_filepath, title=f'{filename}_corrected_displacement')
        else:
            g = g.dropna(subset=['DISPLACEMENT [um]'])
            displacement = g['DISPLACEMENT [um]']
            if save_to_filepath and save_to_filepath != 'ipynb':
                bokeh.io.output_file(save_to_filepath, title=f'{filename}_displacement')

        time = g['FRAME']*dt
        if len(time)>10:
            p.circle(x=time, y=displacement, alpha=0.5, color=color, legend_label=f'{track}')
            p.legend.click_policy = 'hide'
    if save_to_filepath=='ipynb':
        bokeh.io.show(p)
    else:
        bokeh.io.save(p)


def plot_displacement_force_ratio(filename: str, df: pd.DataFrame, dt: int, first_pulse: int, t_on: int, t_off: int,save_to_filepath: str) -> None:
    p = bokeh.plotting.figure(
            frame_width = 400,
            frame_height = 300,
            x_axis_label='Time (s)',
            y_axis_label='Displacement / force (um/pN)',
            title=filename,
            )
    colors = cc.b_glasbey_category10

    magnet_pulses = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on)] for j in range(20)])

    for color, (track, g) in zip(colors, df.groupby('TRACK_ID')):
        for magnet_pulse in magnet_pulses:
            if substracted_background:
                g = g.dropna(subset=['CORRECTED DISPLACEMENT [um]'])
                displacement_force_ratio = g.loc[g['FRAME'].isin(magnet_pulse), 'CORRECTED DISPLACEMENT [um]']/g.loc[g['FRAME'].isin(magnet_pulse), 'FORCE [pN]']
                if save_to_filepath and save_to_filepath != 'ipynb':
                    bokeh.io.output_file(save_to_filepath, title=f'{filename}_corrected_viscosity_fit')
            else:
                g = g.dropna(subset=['DISPLACEMENT [um]'])
                displacement_force_ratio = g.loc[g['FRAME'].isin(magnet_pulse), 'DISPLACEMENT [um]']/g.loc[g['FRAME'].isin(magnet_pulse), 'FORCE [pN]']
                if save_to_filepath and save_to_filepath != 'ipynb':
                    bokeh.io.output_file(save_to_filepath, title=f'{filename}_viscosity_fit')
            time = g.loc[g['FRAME'].isin(magnet_pulse), 'FRAME']*dt
            if len(time) < 3:
                continue

            p.circle(x=time, y=displacement_force_ratio, alpha=0.5, color=color, legend_label=f'{track}')
            f = lambda x, *p: p[0]*x + p[1]
            popt, pcov = curve_fit(f, time, displacement_force_ratio, p0=[0.1, 10], nan_policy='raise')
            x_fit = np.linspace(min(time), max(time), 30)
            y_fit = f(x_fit, *popt)
            eff_viscosity = 1000/(popt[0]*6*np.pi*1.4) #mPas
            df.loc[(df['FRAME'].isin(magnet_pulse)) & (df['TRACK_ID']==track), 'EFF_VISCOSITY']=eff_viscosity
            p.line(x=x_fit, y=y_fit, alpha=0.3, line_width=2, color='black', legend_label='viscosity fit')
    p.circle(x=time[df['MAGNET_STATUS']==1], y=displacement_force_ratio[df['MAGNET_STATUS']==1], alpha=0.5, color='green')

    p.legend.click_policy = 'hide'
    if save_to_filepath=='ipynb':
        bokeh.io.show(p)
    else:
        bokeh.io.save(p)


def plot_viscoelastic_responce(filename: str, df: pd.DataFrame, dt: int, first_pulse: int, t_on: int, t_off: int,save_to_filepath: str, fit='single') -> pd.DataFrame:
    '''
    Plot displacement of each bead in a compact way: plotting consecutive pulses into the same region. Every pulse starts at t=0. This will allow for fitting viscoelastic models.  

    Parameter \'fit\' can either be \'single\' (for fitting every single curve) or \'batch\' (for fitting all the tracks from the bead together).
    '''
    
    # df_viscoelastic_response = pd.DataFrame(columns=['FILENAME', 'TRACK_IDX', 'PULSE_START_FRAME', 'MAGNET_STATUS','MT_STATUS', 'VISCOEL_PARAMS'])
    # divide tracks and cut the pulses
    for track_idx in df['TRACK_ID'].unique():
        if len(df[df['TRACK_ID']==track_idx]) < 2*(t_on+t_off):
            continue
        p = bokeh.plotting.figure(
            frame_width = 400,
            frame_height = 300,
            x_axis_label='time (s)',
            y_axis_label='displacement/force (um/pN)',
            title=filename
        )
        
        N_pulses = np.max(df['FRAME'])//(t_on+t_off)
        magnet_pulses = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on)] for j in range(N_pulses+1)])

        data_x_rising_all = []
        data_y_rising_all = []
        data_x_relaxing_all = []
        data_y_relaxing_all = []

        for magnet_pulse in magnet_pulses:
            time = [dt*i for i in range(t_on+t_off)]

            df_track = df.loc[(df["TRACK_ID"]==track_idx)&(df['FRAME'].isin([i for i in range(int(magnet_pulse[0]), int(magnet_pulse[0]+t_on+t_off))]))]
            # phase = df_track['PHASE'][0]
            # print(phase)

            if len(df_track) < t_on+t_off:
                continue

            displacement = (df_track['DISTANCE [um]'].values[0]-df_track['DISTANCE [um]'].values)/df_track['FORCE [pN]'].values
            data_x_rising = np.asarray(time[:t_on])
            data_y_rising = np.asarray(displacement[:t_on])
            data_x_relaxing = np.asarray(time[t_on-1:])
            data_y_relaxing = np.asarray(displacement[t_on-1:])

            # plot data
            p.line(x=data_x_rising, y=data_y_rising, alpha=0.5, color='green', legend_label='Magnet ON')
            p.line(x=data_x_relaxing, y=data_y_relaxing, alpha=0.5, legend_label='Magnet OFF')
            p.circle(x=data_x_rising, y=data_y_rising, alpha=0.5, color='green', legend_label='Magnet ON')
            p.circle(x=data_x_relaxing, y=data_y_relaxing, alpha=0.5, legend_label='Magnet OFF')

            # plot individual fits
            data_x_rising_fit, data_y_rising_fit, popt_rising, pcov_rising = fit_jeffreys_model(data_x_rising, data_y_rising, 'rising')
            data_x_relaxing_fit, data_y_relaxing_fit, popt_relaxing, pcov_relaxing = fit_jeffreys_model(data_x_relaxing, data_y_relaxing, 'relaxing')

            if all(popt_rising) and fit_sufficient(popt_rising, pcov_rising) and MSE(data_x_rising, data_y_rising, popt_rising)<0.0005:
                p.line(x=data_x_rising_fit, y=data_y_rising_fit, alpha=0.5, color='black', legend_label='Magnet ON')
                data_x_rising_all.extend(data_x_rising)
                data_y_rising_all.extend(data_y_rising)
                # new_rew = {'FILENAME': filename, 
                #        'TRACK_IDX': track_idx, 
                #        'PULSE_START_FRAME': magnet_pulse[0], 
                #        'MAGNET_STATUS': 1,
                #        'MT_STATUS': phase, 
                #        'VISCOEL_PARAMS': popt_rising}

            if all(popt_relaxing) and fit_sufficient(popt_relaxing, pcov_relaxing) and MSE(data_x_relaxing, data_y_relaxing, popt_relaxing)<0.0005:  
                p.line(x=data_x_relaxing_fit, y=data_y_relaxing_fit, alpha=0.5, color='black', legend_label='Magnet OFF')
                data_x_relaxing_all.extend(data_x_relaxing)
                data_y_relaxing_all.extend(data_y_relaxing)



        p.legend.click_policy = 'hide'
        
        if save_to_filepath=='ipynb':
            bokeh.io.show(p)
        else:
            if not os.path.isdir(f"{save_to_filepath}dispacement_force_curves/"):
                os.mkdir(f"{save_to_filepath}dispacement_force_curves/")
            bokeh.io.output_file(f"{save_to_filepath}dispacement_force_curves/displacement_force_ratio_{filename}_track_{track_idx}.html", title='displacement/force')
            bokeh.io.save(p)


def fit_sufficient(popt, pcov) -> bool:
    '''
    Test if the fit is good enough. Te cut off is when error of any parameter is larger than the value of the parameter. 
    '''
    perr = np.sqrt(np.diag(pcov))
    for i in range(len(popt)):
        if popt[i]/perr[i] < 1: # retative error should not be larger than 1. 
            return False
    # print(popt, perr)
    return True

def track_good_enough(data_x, data_y) -> bool:
    '''
    Filter for individual tracks. For now, empty. 
    '''
    return True


def MSE(data_x: np.ndarray, data_y: np.ndarray, fit_popt: np.ndarray) -> float:
    if len(fit_popt) == 3:
        data_fit = jeffreys_model_rising(data_x, *fit_popt)
    elif len(fit_popt) == 2:
        data_fit = jeffreys_model_relaxing(data_x, *fit_popt)
    mse = 1/len(data_y)*sum([data_y[i]**2-data_fit[i]**2 for i in range(len(data_y))])
    return mse



