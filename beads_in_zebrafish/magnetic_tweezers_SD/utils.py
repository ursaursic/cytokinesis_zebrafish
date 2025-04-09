'''
Ursa Ursic, last updated: 4.4.2024

Utility file for all the bead motion analysis. 
'''
import pandas as pd
import numpy as np
import os
import cv2
from scipy.optimize import curve_fit, Bounds
from scipy.signal import find_peaks
import colorcet as cc
from typing import Callable
import matplotlib.pyplot as plt

use_matplotlib = True

#######
# define force calibration parameters
#######
force_calibration_params = {
    1000: [1.74898497e+03, 2.76024353e-02, 1.26564525e+02, 5.08675459e-03],
    '1000': [1.74898497e+03, 2.76024353e-02, 1.26564525e+02, 5.08675459e-03],
}


window = 1/2 # the window for background subtraction

################################################################################
# FUNCTIONS FOR add_info_to_df
################################################################################

def plot_tip(df_tracks: pd.DataFrame, tip_end: [int, int], save_img_to_path: str, pix_size: float) -> None:
    '''
    Plot where the tip is in relationship to all the tracks. Save the images for sanity check.
    '''

    # plot in pixels
    tracks_x, tracks_y = df_tracks['POSITION_X'].values, df_tracks['POSITION_Y'].values
    plt.figure()
    plt.plot(tip_end[0], tip_end[1], 'ro')
    plt.plot(tracks_x/pix_size, tracks_y/pix_size, 'g.', alpha =0.2)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=np.max((np.max(tracks_y)/pix_size, 2*tip_end[1])))
    plt.savefig(save_img_to_path)
    plt.close()


def calculate_distance_from_tip(df: pd.DataFrame, tip: list, pix_size: float) -> None:
    '''
    Calculates distance from the tip end point (tip_point is in pixels) to bead and adds this to the dataframe df. The tracks POSITION_X and POSITION_Y are in um. 
    '''
    if len(tip) == 2:
        df['DISTANCE [um]'] = np.sqrt((df['POSITION_X']-tip[0]*pix_size)**2+(df['POSITION_Y']-tip[1]*pix_size)**2)


def add_magnet_status(df: pd.DataFrame, magnet_info: list[int]) -> None:
    '''
    Add the information weather the magnemagnet is on or off. 
    '''
    first_pulse, last_pulse, t_on, t_off = magnet_info
    # number of pulses to take into account:
    N_pulses = (last_pulse - first_pulse)//(t_on+t_off) + 1

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
    df['FORCE [pN]'] = a1*np.exp(-df['DISTANCE [um]']*k1)+a2*np.exp(-df['DISTANCE [um]']*k2)
    df.loc[df['MAGNET_STATUS']==0, 'FORCE [pN]'] = 0


def add_calculated_displacement(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function substracts the signal from the background (so that we get increasing displacement after each first point of the new pulse). It creates a new column in the df dataframe with the displacement values after each pulse. 

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


def add_flow_slope(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function checks out the bead tracks. It takes into account the last window*length of the OFF phase before the new period starts. It calculates the slopes where possible and adds them into the data frame.  
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
            xdata = track.loc[track['PULSE_NUMBER']==pulse, 'FRAME'].values[-int(window*magnet_off_length):]
            ydata = track.loc[track['PULSE_NUMBER']==pulse, 'DISTANCE [um]'].values[-int(window*magnet_off_length):]
            f = lambda x, *p: p[0]*x + p[1]
            popt, pcov = curve_fit(f, xdata, ydata, p0=[0, 10])
            # if (np.sqrt(pcov[1][1])/popt[1] < 1) & (np.sqrt(pcov[0][0])/popt[0] < 1) & (MSE(xdata, ydata, f, popt) < 0.5):
            df.loc[(df["TRACK_ID"]==idx) & (track['PULSE_NUMBER']==pulse), ['CORRECTION_k', 'CORRECTION_k_ERR', 'CORRECTION_N', 'CORRECTION_N_ERR']] = [popt[0], pcov[0][0], popt[1], pcov[1][1]]
    return df


def MSE(data_x: np.ndarray, data_y: np.ndarray, f: Callable, fit_popt: np.ndarray) -> float:
    data_fit = f(data_x, *fit_popt)
    mse = 1/len(data_y)*sum([data_y[i]**2-data_fit[i]**2 for i in range(len(data_y))])
    return mse


################################################################################
# FUNCTIONS FOR CALCULATING MATERIAL PROPERTIES
################################################################################

def calculate_model_independedt_params(pulse: pd.DataFrame, avg_force: float) -> tuple:
    rising_phase = pulse.loc[pulse['MAGNET_STATUS']==1, 'CORRECTED DISPLACEMENT [um]'].values
    relaxing_phase = pulse.loc[pulse['MAGNET_STATUS']==0, 'CORRECTED DISPLACEMENT [um]'].values
    
    rising_dif = rising_phase[-1]-rising_phase[0]
    relaxing_dif = relaxing_phase[-1]-relaxing_phase[0]

    rising_dif_norm = rising_dif/avg_force
    rising_dif_norm_inverse = 1/rising_dif_norm

    return rising_dif, relaxing_dif, rising_dif_norm, rising_dif_norm_inverse


def r_squared(ydata, yfit):
    ss_res = np.sum((ydata - yfit) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    R_sq = 1- ss_res/ss_tot
    return R_sq


def jeff_full(t, k, eta_1, eta_2, F_0, t_1) -> np.ndarray:
    '''
    fit Jeffrey's rising and relaxing phases to data. Jefferey's model is composed of a sping with an elastic constant k, parralely bound to a dashpod with viscosity eta_1. This component is also sequentially bound to another dashpod with viscosity eta_2. The model is fitted with a "rising pahse", where we assume a constant force at each time step and a "relaxing phase" where the force is 0, but the system is relaxing after the force was turned off. 
          |--- dashpod (eta_1)--|
    |-----|                     |---- dashpod (eta 2) -----> F
          |--- spring (k) ------|
    '''
    a = 1 - 1 / ((eta_2 / (k * t_1)) * (1 - np.exp(- k* t_1 / eta_1)) + 1) 

    # rising
    x_1 = F_0 / k * (1 - np.exp(-k * t / eta_1)) + F_0 * t / eta_2

    # relaxing
    x_2 = (F_0 / k * (1 - np.exp(-k * t_1 / eta_1)) + F_0 * t_1 / eta_2) * (a * np.exp(-(t-t_1) * k / eta_1) + (1-a))

    # combined
    x = list(x_1[:len(t[t<t_1])]) + list(x_2[len(t[t<t_1]):])
    return x


def KV_full(t, k, eta, F_0, t_1) -> np.ndarray:
    '''
    fit Kelvin-Voigt rising and relaxing phases to data. 
          |--- dashpod (eta)----|
    |-----|                     | -----> F
          |--- spring (k) ------|
    '''

    # rising
    x_1 = F_0 / k * (1 - np.exp(-k * t / eta))

    # relaxing
    x_2 = F_0 / k * (1 - np.exp(-k * t_1 / eta)) * (np.exp(-(t-t_1) * k / eta))

    # combined
    x = list(x_1[:len(t[t<t_1])]) + list(x_2[len(t[t<t_1]):])
    return x


def get_fit_jeff_full(xfit, xdata, ydata, F_0, t_1, sigma=None) -> np.ndarray:
    jeff_full_for_fit = lambda t, k, eta_1, eta_2: jeff_full(t, k, eta_1, eta_2, F_0, t_1)

    popt, pcov = curve_fit(jeff_full_for_fit, xdata, ydata, p0=[30, 40, 100], bounds=([0.5, 0.5, 0.5], [10e5, 10e5, 10e5]), sigma=sigma, nan_policy='omit') # lower bounds to prevent exponent to overflow

    yfit = jeff_full_for_fit(xfit, *popt)

    return yfit, popt, pcov


def get_fit_KV_full(xfit, xdata, ydata, F_0, t_1, sigma=None) -> np.ndarray:
    KV_full_for_fit = lambda t, k, eta: KV_full(t, k, eta, F_0, t_1)

    popt, pcov = curve_fit(KV_full_for_fit, xdata, ydata, p0=[100, 100], bounds=([0.5, 0.5], [10e5, 10e5]), sigma=sigma, nan_policy='omit')  # lower bounds to prevent exponent to overflow

    yfit = KV_full_for_fit(xfit, *popt)

    return yfit, popt, pcov


def calculate_KV_fit_params(time_data, displacement_full, avg_force, t_1, dt, sigma, plot=False):
    time_fit = np.linspace(0, len(time_data) * dt, 100)

    displacement_fit_KV, popt_KV, pcov_KV = get_fit_KV_full(time_fit, time_data, displacement_full, avg_force, t_1, sigma)

    params = ['k', 'eta']

    k_KV, eta_KV = popt_KV
    k_KV_err, eta_KV_err = [np.sqrt(pcov_KV[0][0]), np.sqrt(pcov_KV[1][1])]
    yfit_KV = KV_full(np.array(time_data), k_KV, eta_KV, avg_force, t_1)
    R_sq_KV = r_squared(np.array(displacement_full), np.array(yfit_KV))

    text_KV = ''
    for i in range(len(popt_KV)):
        text_KV += f"{params[i]}: {round(popt_KV[i], 2)} +/- {round(np.sqrt(pcov_KV[i][i]), 2)}\n"
    text_KV += f"$R^2$: {round(R_sq_KV, 2)}\n"
    
    if plot:
        plt.plot(time_fit, displacement_fit_KV, 'r--', label = f'KV fit: {text_KV}')

    return k_KV, eta_KV, k_KV_err, eta_KV_err, R_sq_KV


def calculate_Jeff_fit_params(time_data, displacement_full, avg_force, t_1, dt, sigma, plot=False):
    params = ['k', 'eta_1', 'eta_2']
    
    # find fit parameters
    time_fit = np.linspace(0, len(time_data) * dt, 100)
    displacement_fit, popt, pcov = get_fit_jeff_full(time_fit, time_data, displacement_full, avg_force, t_1, sigma)

    k, eta_1, eta_2 = popt
    k_err, eta_1_err, eta_2_err = [np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])]
    yfit = jeff_full(np.array(time_data), k, eta_1, eta_2, avg_force, t_1)
    R_sq = r_squared(np.array(displacement_full), np.array(yfit))

    text = ''
    for i in range(len(popt)):
        text += f"{params[i]}: {round(popt[i], 2)} +/- {round(np.sqrt(pcov[i][i]), 2)}\n"
    text += f"$R^2$: {round(R_sq, 2)}\n"

    if plot:
        plt.plot(time_fit, displacement_fit, 'k--', label=f'fit: {text}')

    return k, eta_1, eta_2, k_err, eta_1_err, eta_2_err, R_sq


def check_differentiable(time_data, k, eta_1, eta_2, avg_force, t_1, dt, plot=False):
    'Check if fit is (practically) differentiable. Plot if needed.'
    fit_divisible = True
    time_fit = np.linspace(0, len(time_data) * dt, 100)
    displacement_fit = jeff_full(time_fit, k, eta_1, eta_2, avg_force, t_1)

    derivative = np.diff(displacement_fit)
    peaks, properties = find_peaks(-derivative, prominence=0.2, width=0.1)
    if len(peaks) > 0:
        if properties['widths'][0] < 1:
            if plot:
                plt.plot(time_fit[peaks], derivative[peaks], 'o', color='green', label = 'peaks')
            fit_divisible = False

    if plot:
        plt.plot(time_fit[:-1], derivative, 'r-', label = 'derivative')
        plt.title('Check if the fit is divisible')
        plt.xlabel('Time (s)')
        plt.ylabel('Derivative of displacement')
        plt.show()

    return fit_divisible



################################################################################
# FUNCTIONS FOR PLOTTING
################################################################################

def plot_trajectories(filename: str, df: pd.DataFrame, comments: str, save_to_filepath: str='ipynb', show_background_fit=False):
    if len(df) < 10:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_title(f'{filename}\ncomments: {comments}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance from tip (um)')

    # Scatter plot for Magnet ON and OFF
    df_on = df[df["MAGNET_STATUS"] == 1]
    df_off = df[df["MAGNET_STATUS"] == 0]

    ax.scatter(df_on['FRAME'], df_on['DISTANCE [um]'], alpha=0.5, color='green', s=5, label='Magnet ON')
    ax.scatter(df_off['FRAME'], df_off['DISTANCE [um]'], alpha=0.5, color='blue', s=5, label='Magnet OFF')

    if show_background_fit:
        for idx in df['TRACK_ID'].unique():
            track = df[df["TRACK_ID"] == idx]
            pulses = track['PULSE_NUMBER'].unique()
            period_length = max([
                len(track[track["PULSE_NUMBER"] == pulse]['FRAME'].values) for pulse in pulses
            ])

            for pulse in pulses:
                pulse_track = track[track["PULSE_NUMBER"] == pulse]
                if len(pulse_track) < 1/4 * period_length:
                    continue

                magnet_off = pulse_track[pulse_track["MAGNET_STATUS"] == 0]
                magnet_off_length = len(magnet_off)
                if magnet_off_length == 0:
                    continue

                xdata = pulse_track["FRAME"].values[-magnet_off_length:]
                ydata = pulse_track["DISTANCE [um]"].values[-magnet_off_length:]
                popt = pulse_track[["CORRECTION_k", "CORRECTION_N"]].values[0]
                if popt.size != 0 and not np.isnan(popt).any():
                    k, N = popt
                    f = lambda x: k * x + N

                    # Plot data used for fitting
                    xdata_window = xdata[-int(window * magnet_off_length):]
                    ydata_window = ydata[-int(window * magnet_off_length):]
                    ax.scatter(xdata_window, ydata_window, alpha=0.3, s=5, color='red', label='Data for drift fit')

                    # Plot fitted line
                    x_fit = np.linspace(min(xdata), max(xdata), 30)
                    y_fit = f(x_fit)
                    ax.plot(x_fit, y_fit, '--', alpha=0.8, linewidth=1, color='black', label='Drift fit')

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    plt.tight_layout()
    if save_to_filepath == 'ipynb':
        plt.show()
    else:
        plt.savefig(save_to_filepath, dpi=300)
        plt.close()


def plot_displacement(filename: str, df: pd.DataFrame, comments: str, save_to_filepath: str) -> None:
    if len(df['CORRECTED DISPLACEMENT [um]'].dropna()) < 10:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'{filename}\ncomments: {comments}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (um)')

    # Estimate time step dt
    dt = np.average([
        df['POSITION_T'].values[i] / df['FRAME'].values[i]
        for i in range(len(df)) if df['POSITION_T'].values[i] != 0
    ])

    colors = cc.b_glasbey_category10
    for color, (track, g) in zip(colors, df.groupby('TRACK_ID')):
        if 'CORRECTED DISPLACEMENT [um]' in df.columns:
            g = g.dropna(subset=['CORRECTED DISPLACEMENT [um]'])
            displacement = g['CORRECTED DISPLACEMENT [um]']
        else:
            g = g.dropna(subset=['DISPLACEMENT [um]'])
            displacement = g['DISPLACEMENT [um]']

        time = g['FRAME'] * dt

        if len(time) > 10:
            ax.plot(time, displacement, '.-', markersize=3, alpha=0.5, color=color, label=f'{track}')

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='right', fontsize='small')

    plt.tight_layout()
    if save_to_filepath == 'ipynb':
        plt.show()
    else:
        plt.savefig(save_to_filepath, dpi=300)
        plt.close()
