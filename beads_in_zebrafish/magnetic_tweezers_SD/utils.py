'''
Ursa Ursic, last updated: 4.4.2024

Utility file for all the bead motion analysis. 
'''
import pandas as pd
import numpy as np
import os
import cv2
from scipy.optimize import curve_fit, Bounds
import colorcet as cc
from typing import Callable

import bokeh.io
import bokeh.plotting
import bokeh.models

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

################################################################################
# FUNCTIONS FOR add_info_to_df
################################################################################

def find_tip(filepath_tip: str, threshold_tip: int, save_img_to_path: str, endpoint: bool) -> list:
    img = cv2.imread(filepath_tip, cv2.IMREAD_UNCHANGED)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    tip_mask = blurred_img < threshold_tip
    tip_mask = np.array(tip_mask.astype(int))
    tip_outline = np.array([[0, 0]])
    tip_end = [0, 0]

    boundaries_for_tip = np.array([[img.shape[0]//5, 4*img.shape[0]//5], [1, img.shape[1]*1//4]] , dtype=int)

    for i in range(boundaries_for_tip[0, 0], boundaries_for_tip[0, 1]):
        for j in range(boundaries_for_tip[1, 0], boundaries_for_tip[1, 1]):
            if np.sum(tip_mask[i, j] != tip_mask[i-1:i+1, j-1:j+1])==2:
                tip_outline = np.concatenate([tip_outline, [[j, i]]], axis=0)
                if j > tip_end[0]:
                    tip_end = [j, i]

    if tip_outline.shape[0] > 20: 
        if save_img_to_path: 
            filename = os.path.basename(save_img_to_path).split('/')[-1].split('.')[0] 
            p = bokeh.plotting.figure(width=800, height=800, title=f"Yay, we found a tip!!! \n file: {filename}")
            p.title.text_font_size = '16pt'
            p.x_range.range_padding = p.y_range.range_padding = 0
            p.image(image=[img], x=0, y=0, dw=img.shape[0], dh=img.shape[1])
            p.circle(x=tip_outline[:, 0], y=tip_outline[:, 1], color='red')
            p.star(x=tip_end[0], y=tip_end[1], color='cyan', size=15)
            if save_img_to_path=='ipynb':
                bokeh.io.show(p)
            else:
                bokeh.io.export_png(p, filename=f"{save_img_to_path}")
        if endpoint:
            return tip_end
        else: 
            return tip_outline
    else:
        print("No tip found... try another threshold.")
        return False


def calculate_distance_from_tip(df: pd.DataFrame, tip: list) -> None:
    '''
    Calculates distance from the tip end point (tip_point) to bead and adds this to the dataframe df.
    '''
    if len(tip) == 2:
        df['DISTANCE [um]'] = np.sqrt((df['POSITION_X']-tip[0])**2+(df['POSITION_Y']-tip[1])**2)
    else:
        # TODO (find the nearest point to tip)
        pass


def add_magnet_status(df: pd.DataFrame, magnet_info: list[int]) -> None:
    first_pulse, t_on, t_off = magnet_info
    # number of pulses in total
    N_pulses = np.max(df['FRAME'])//(t_on+t_off)
    # Add info about the magnet into the df
    magnet_pulses = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on)] for j in range(N_pulses+1)])
    
    df['MAGNET_STATUS'] = [int(1) if df['FRAME'].values[i] in magnet_pulses else int(0) for i in range(len(df))]
    df['PULSE_NUMBER'] = 0
    for (i, pulse) in enumerate(magnet_pulses):
        df.loc[df['FRAME'].isin(range(pulse[0], pulse[0]+t_on+t_off)), 'PULSE_NUMBER'] = int(i+1)
    

def calculate_force(df: pd.DataFrame, calibration: int | str):
    '''
    Calculates force at the distance of the bead, depending on the force calibration parameters (defined above).
    '''
    a1, k1, a2, k2 = force_calibration_params[calibration]
    df['FORCE [pN]'] = a1*np.exp(df['DISTANCE [um]']/k1)+a2*np.exp(df['DISTANCE [um]']/k2)
    df.loc[df['MAGNET_STATUS']==0, 'FORCE [pN]'] = 0


def add_MT_status(df: pd.DataFrame, MT_info: np.ndarray | str):
    '''
    Add information about the microtubules into the data frame.
    '''
    df['MT_STATUS'] = np.nan
    if len(MT_info) > 1:
        for (i, phase) in enumerate(['m_phase_1', 'i_phase_1', 'm_phase_2', 'i_phase_2']):
            if (MT_info[i*2] != 'na') & (MT_info[i*2+1] != 'na'):
                df.loc[df['FRAME'].isin(range(int(MT_info[i*2]), int(MT_info[i*2+1]))), 'PHASE'] = phase
        
        for i in range(len(df)):
            if type(df['PHASE'].values[i])==str and df['PHASE'].values[i].startswith('i'):
                df['MT_STATUS'].values[i] = int(1)
            elif type(df['PHASE'].values[i])==str and df['PHASE'].values[i].startswith('m'):
                df['MT_STATUS'].values[i] = int(0)


################################################################################
# FUNCTIONS FOR calculate_viscoelastic_responce
################################################################################

def add_flow_slope(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function checks out the bead tracks. It takes into account the last 2/3 of the OFF phase before the new period starts. It calculates the slopes where possible and adds them into the data frame.  
    ---
    df:             dataframe with bead tracks
    ---
    Returns none, because the point is to add a column into the dataframe. 
    '''

    df[['CORRECTION_k', 'CORRECTION_k_ERR', 'CORRECTION_N', 'CORRECTION_N_ERR']] = [np.nan, np.nan, np.nan, np.nan]

    for idx in df['TRACK_ID'].unique():
        track = df[df["TRACK_ID"] == idx]
        for pulse in track['PULSE_NUMBER'].unique():
            if len(track[track['PULSE_NUMBER']==pulse]) <= 15:
                continue
            magnet_off_length = len(track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==0), 'FRAME'])
            xdata = track.loc[track['PULSE_NUMBER']==pulse, 'FRAME'].values[-int(2/3*magnet_off_length):]
            ydata = track.loc[track['PULSE_NUMBER']==pulse, 'DISTANCE [um]'].values[-int(2/3*magnet_off_length):]
            f = lambda x, *p: p[0]*x + p[1]
            popt, pcov = curve_fit(f, xdata, ydata, p0=[-0.1, 100], nan_policy='raise')
            if (np.sqrt(pcov[1][1])/popt[1] < 1) & (np.sqrt(pcov[0][0])/popt[0] < 1) & (MSE(xdata, ydata, f, popt) < 0.5):
                df.loc[(df["TRACK_ID"]==idx) & (track['PULSE_NUMBER']==pulse), ['CORRECTION_k', 'CORRECTION_k_ERR', 'CORRECTION_N', 'CORRECTION_N_ERR']] = [popt[0], pcov[0][0], popt[1], pcov[1][1]]
    return df



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


def fit_sufficient(popt, pcov) -> bool:
    '''
    Test if the fit is good enough. Te cut off is when error of any parameter is larger than the value of the parameter. 
    '''
    perr = np.sqrt(np.diag(pcov))
    for i in range(len(popt)):
        if perr[i] == 0:
            return False
        elif popt[i]/perr[i] < 1: # retative error should not be larger than 1. 
            return False
    return True


def MSE(data_x: np.ndarray, data_y: np.ndarray, f: Callable, fit_popt: np.ndarray) -> float:
    data_fit = f(data_x, *fit_popt)
    mse = 1/len(data_y)*sum([data_y[i]**2-data_fit[i]**2 for i in range(len(data_y))])
    return mse


def add_calculated_displacement(df: pd.DataFrame, subtract_background: bool=False) -> pd.DataFrame:
    '''This function substracts the signal from the background (so that we get increasing displacement after each first point of the new pulse). It creates a new column in the df dataframe with the displacement values after each pulse. 

    ---
    df:             dataframe with bead tracks
    substract_background: calculate displacement considering the background flows. If true, both displacement and corrected displacement will be calculated.
    ---

    Returns none, because the point is to add a column into the dataframe. 
    '''
    df = df.sort_values(by='FRAME')

    # Calculate displacement without background correction
    df['DISPLACEMENT [um]'] = np.nan
    for idx in df['TRACK_ID'].unique():
        track = df[df["TRACK_ID"] == idx]
        for pulse in df['PULSE_NUMBER'].unique():
            data = df.loc[(df['TRACK_ID']==idx) & (df["PULSE_NUMBER"]==pulse), 'DISTANCE [um]'].values
            if len(data)>0:
                displacement = data[0] - data
                df.loc[(df['TRACK_ID']==idx) & (df["PULSE_NUMBER"]==pulse), 'DISPLACEMENT [um]'] = displacement


    df = add_flow_slope(df)
    # Calculate dispolacement with background correction
    df['CORRECTED DISPLACEMENT [um]'] = np.nan

    for idx in df['TRACK_ID'].unique():
        track = df[df["TRACK_ID"] == idx]
        period_length = max([len(track[track["PULSE_NUMBER"]==pulse]['FRAME'].values) for pulse in track['PULSE_NUMBER'].unique()])
        if len(track) < period_length:
            continue
        # background is calculated as accelerated movement
        background_func = lambda t, t_1, x_1, k_1, k_2: x_1 + k_1*(t-t_1) + (k_2-k_1)/(2*(period_length))*(t-t_1)**2

        for pulse in track['PULSE_NUMBER'].unique()[:-1]:
            if len(track[track["PULSE_NUMBER"]==pulse+1]) < 3/4*period_length:
                continue
            popt_1 = track.loc[track["PULSE_NUMBER"]==pulse,  ["CORRECTION_k", "CORRECTION_N"]].values[0]
            popt_2 = track.loc[track["PULSE_NUMBER"]==pulse+1,  ["CORRECTION_k", "CORRECTION_N"]].values[0]
            t_0 = track.loc[track["PULSE_NUMBER"]==pulse+1, 'FRAME'].values[0]
            
            if not (np.isnan(popt_1).any() or np.isnan(popt_2).any()):
                k_1, n_1 = popt_1
                k_2, _ = popt_2
                x_1 = k_1*t_0 + n_1
                time = track[track["PULSE_NUMBER"]==pulse+1]["FRAME"].values
                data = track[track["PULSE_NUMBER"]==pulse+1]["DISTANCE [um]"].values
                corrected_data = background_func(time, time[0], x_1, k_1, k_2) - data 
                corrected_data -= corrected_data[0] 
                df.loc[(df['TRACK_ID']==idx) & (df["PULSE_NUMBER"]==pulse+1), 'CORRECTED DISPLACEMENT [um]'] = corrected_data
    return df


################################################################################
# FUNCTIONS FOR PLOTTING
################################################################################

def plot_trajectories(filename: str, df: pd.DataFrame, comments: str, save_to_filepath: str, show_background_fit=False) -> None:
    if len(df) < 10:
        return None
    p = bokeh.plotting.figure(
    frame_width = 600,
    frame_height = 400,
    x_axis_label='Frame',
    y_axis_label='Distance from tip (um)',
    title=f'{filename}\ncomments: {comments}'
    )

    p.add_layout(bokeh.models.Legend(), 'right')

    source_on = bokeh.models.ColumnDataSource(df[df["MAGNET_STATUS"]==1])
    source_off = bokeh.models.ColumnDataSource(df[df["MAGNET_STATUS"]==0])
    p.circle(source=source_on, x='FRAME', y='DISTANCE [um]', alpha=0.5, color='green', legend_label='Magnet ON')
    p.circle(source=source_off, x='FRAME', y='DISTANCE [um]', alpha=0.5, legend_label='Magnet OFF')

    if show_background_fit:
        for idx in df['TRACK_ID'].unique():
            track = df[df["TRACK_ID"] == idx]
            period_length = max([len(track[track["PULSE_NUMBER"]==pulse]['FRAME'].values) for pulse in track['PULSE_NUMBER'].unique()])

            for pulse in track["PULSE_NUMBER"].unique():
                if len(track[track["PULSE_NUMBER"]==pulse]) < 1/4*period_length:
                    continue
                
                magnet_off_length = len(track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==0), 'FRAME'])
                xdata = track[track["PULSE_NUMBER"]==pulse]["FRAME"].values[-magnet_off_length:]
                ydata = track[track["PULSE_NUMBER"]==pulse]["DISTANCE [um]"].values[-magnet_off_length:]
                popt = track.loc[track["PULSE_NUMBER"]==pulse,  ["CORRECTION_k", "CORRECTION_N"]].values[0]
                if popt.size != 0 and not np.isnan(popt).any():
                    f = lambda x, k, N: k*x + N  
                    p.circle(x=xdata[-int(2/3*magnet_off_length):], y=ydata[-int(2/3*magnet_off_length):], alpha=0.3, size=2, color='red', legend_label='Data for drift fit')
                    x_fit = np.linspace(min(xdata), max(xdata), 30)
                    y_fit = f(x_fit, *popt)
                    p.line(x=x_fit, y=y_fit, alpha=0.3, line_width=2, color='black', legend_label='Drift fit')

    p.legend.click_policy = 'hide'
    if save_to_filepath=='ipynb':
        bokeh.io.show(p)
    else:
        bokeh.io.export_png(p, filename=f"{save_to_filepath}")

def plot_displacement(filename: str, df: pd.DataFrame, comments: str, save_to_filepath: str) -> None:
    if len(df) < 10:
        return None
    p = bokeh.plotting.figure(
            frame_width = 600,
            frame_height = 400,
            x_axis_label='Time (s)',
            y_axis_label='Displacement (um)',
            title=f'{filename}\ncomments: {comments}'
            )
    
    p.add_layout(bokeh.models.Legend(), 'right')
    colors = cc.b_glasbey_category10
    
    dt = np.average([df['POSITION_T'].values[i]/df['FRAME'].values[i] for i in range(len(df)) if df['POSITION_T'].values[i] != 0])
    # print(f'Sanity check: dt = {dt}')

    for color, (track, g) in zip(colors, df.groupby('TRACK_ID')):
        if 'CORRECTED DISPLACEMENT [um]' in df.columns:
            g = g.dropna(subset=['CORRECTED DISPLACEMENT [um]'])
            displacement = g['CORRECTED DISPLACEMENT [um]']
        else:
            g = g.dropna(subset=['DISPLACEMENT [um]'])
            displacement = g['DISPLACEMENT [um]']

        time = g['FRAME']*dt
        if len(time)>10:
            p.circle(x=time, y=displacement, alpha=0.5, color=color, legend_label=f'{track}')
            p.legend.click_policy = 'hide'
    if save_to_filepath=='ipynb':
        bokeh.io.show(p)
    else:
        bokeh.io.export_png(p, filename=f"{save_to_filepath}")


def calculate_displacement_force_ratio(track: pd.DataFrame, pulse: int) -> list[np.ndarray, np.ndarray]:
    distance_magnet_on = track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==1), 'DISTANCE [um]'].values
    distance_magnet_off = track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==0), 'DISTANCE [um]'].values
    force_magnet_on = track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==1), 'FORCE [pN]'].values

    displacement_force_magnet_on = (distance_magnet_on[0] - distance_magnet_on)/force_magnet_on
    displacement_force_magnet_off = (distance_magnet_on[0]-distance_magnet_off)/np.average(force_magnet_on)
    # displacement_force_magnet_off /=displacement_force_magnet_off[0]  # normalization ? 
    return displacement_force_magnet_on, displacement_force_magnet_off


def set_time_data(track: pd.DataFrame, pulse: int, dt: float) -> list[np.ndarray, np.ndarray]:
    time_on = dt*track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==1), 'FRAME'].values
    time_off = dt*track.loc[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==0), 'FRAME'].values

    time_off -= time_on[0]
    time_on -= time_on[0]
    return time_on, time_off


def plot_viscoelastic_responce(filename: str, df: pd.DataFrame, comments: str):
    if len(df) < 10:
        return None
    p = bokeh.plotting.figure(
        frame_width = 600,
        frame_height = 400,
        x_axis_label='time (s)',
        y_axis_label='displacement/force (um/pN)',
        title=f'{filename}\ncomments: {comments}'
    )
    # mytext = bokeh.models.Label(x=70, y=70, text=f'{comments}')

    # p.add_layout(mytext, 'right')
    p.add_layout(bokeh.models.Legend(), 'right')
    return p


def add_data_to_plot(p, time_on, time_off, displacement_force_magnet_on, displacement_force_magnet_off, MT_status) -> None:
    # plot data
    p.line(x=time_on, y=displacement_force_magnet_on, alpha=0.5, color='green', legend_label='Magnet ON')
    p.line(x=time_off, y=displacement_force_magnet_off, alpha=0.5, legend_label='Magnet OFF')
    p.circle(x=time_on, y=displacement_force_magnet_on, alpha=0.5, color='green', legend_label='Magnet ON')
    p.circle(x=time_off, y=displacement_force_magnet_off, alpha=0.5, legend_label='Magnet OFF')

    if MT_status == 1 or MT_status == 'y':
        MT_label = 'MT'
    elif MT_status == 0 or MT_status == 'n':
        MT_label = 'noMT'
    elif MT_status == 'CHX':
        MT_label = 'CHX'
    else:
        MT_label = '?'
    MT_label_on = bokeh.models.Label(x=time_on[-1], y=displacement_force_magnet_on[-1], text_alpha=0.5, text=MT_label)
    MT_label_off = bokeh.models.Label(x=time_off[-1], y=displacement_force_magnet_off[-1], text_alpha=0.5, text=MT_label)
    
    p.add_layout(MT_label_on)
    p.add_layout(MT_label_off)


def add_fit_to_plot(p, data_x_fit, data_y_fit, magnet_status: str) -> None:
    p.line(x=data_x_fit, y=data_y_fit, alpha=0.5, color='black', legend_label=magnet_status)


def save_plot(p, filename, track_idx, save_to_filepath):
    p.legend.click_policy = 'hide'

    if save_to_filepath=='ipynb':
        bokeh.io.show(p)
    else:
        if not os.path.isdir(f"{save_to_filepath}/dispacement_force_curves/"):
            os.mkdir(f"{save_to_filepath}/dispacement_force_curves/")
        bokeh.io.export_png(p, filename=f"{save_to_filepath}/dispacement_force_curves/{filename}_{track_idx}.png")


def calculate_fit_parameters(filename: str, df: pd.DataFrame, comments: str, df_results: pd.DataFrame, save_plot_to_filepath: str) -> pd.DataFrame:
    dt = np.average([df['POSITION_T'].values[i]/df['FRAME'].values[i] for i in range(len(df)) if df['POSITION_T'].values[i] != 0])
    
    idx = 0
    for track_idx in df['TRACK_ID'].unique():
        track = df[df["TRACK_ID"]==track_idx]
        period_length = max([len(track[track["PULSE_NUMBER"]==pulse]['FRAME'].values) for pulse in track['PULSE_NUMBER'].unique()])
        if len(df[df['TRACK_ID']==track_idx]) < 2*period_length: 
            continue
        
        if save_plot_to_filepath:
            p = plot_viscoelastic_responce(filename, df, comments)
        
        for pulse in track['PULSE_NUMBER'].unique():
            if len(track[track['PULSE_NUMBER']==pulse]) < period_length or len(track[(track['PULSE_NUMBER']==pulse)&(track['MAGNET_STATUS']==1)]) < 5:
                continue

            # Define time and displacement
            time_on, time_off = set_time_data(track, pulse, dt)
            displacement_force_magnet_on, displacement_force_magnet_off = calculate_displacement_force_ratio(track, pulse)
            MT_status = track.loc[track['PULSE_NUMBER']==pulse, 'MT_STATUS'].values[0]
            if save_plot_to_filepath:
                add_data_to_plot(p, time_on, time_off, displacement_force_magnet_on, displacement_force_magnet_off, MT_status)

            # Calculate fit
            data_x_rising_fit, data_y_rising_fit, popt_rising, pcov_rising = fit_jeffreys_model(time_on, displacement_force_magnet_on, 'rising')
            data_x_relaxing_fit, data_y_relaxing_fit, popt_relaxing, pcov_relaxing = fit_jeffreys_model(time_off, displacement_force_magnet_off, 'relaxing')

            # Add fit parameters to the result dataframe
            new_line = {'FILENAME': filename, 
                        'TRACK_IDX': track_idx, 
                        'PULSE_NUMBER': pulse, 
                        'MT_STATUS': MT_status,
                        'VISCOEL_PARAMS_RISING_k': np.nan,
                        'VISCOEL_PARAMS_RISING_eta1': np.nan,
                        'VISCOEL_PARAMS_RISING_eta2': np.nan,
                        'VISCOEL_PARAMS_RELAXING_a': np.nan,
                        'VISCOEL_PARAMS_RELAXING_tau': np.nan,
                        'COMMENTS': str(comments)
                        }

            flag = [False, False]
            if all(popt_rising) and fit_sufficient(popt_rising, pcov_rising) and MSE(time_on, displacement_force_magnet_on, jeffreys_model_rising, popt_rising) < 0.0005:
                new_line['VISCOEL_PARAMS_RISING_k'] = popt_rising[0]
                new_line['VISCOEL_PARAMS_RISING_eta1'] = popt_rising[1]
                new_line['VISCOEL_PARAMS_RISING_eta2'] = popt_rising[2]
                if save_plot_to_filepath:
                    add_fit_to_plot(p, data_x_rising_fit, data_y_rising_fit, 'Magnet ON')
                flag[0] = True

            if all(popt_relaxing) and fit_sufficient(popt_relaxing, pcov_relaxing) and MSE(time_off, displacement_force_magnet_off, jeffreys_model_relaxing, popt_relaxing) < 0.0005:  
                new_line['VISCOEL_PARAMS_RELAXING_a'] = popt_relaxing[0]
                new_line['VISCOEL_PARAMS_RELAXING_tau'] = popt_relaxing[1]
                if save_plot_to_filepath:
                    add_fit_to_plot(p, data_x_relaxing_fit, data_y_relaxing_fit, 'Magnet OFF')
                flag[1] = True

            if any(flag):
                df_results = pd.concat([df_results,pd.DataFrame(new_line, index=[idx])])
                idx += 1
        if save_plot_to_filepath:
            save_plot(p, filename, track_idx, save_plot_to_filepath)
    return df_results

