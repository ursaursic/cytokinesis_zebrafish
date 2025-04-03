'''
Ursa Ursic
last update: 22.5.2024

Utility file for calibrating magnetic tweezers in the Brugues lab.
'''

import pandas as pd
import numpy as np
import os
import cv2
from scipy.optimize import curve_fit
import colorcet as cc

import xml.etree.ElementTree as ET
import pandas as pd
from io import StringIO

import bokeh.io
import bokeh.plotting
import bokeh.models

import matplotlib.pyplot as plt
import matplotlib as mpl


def xml_to_df(trackmate_file: str) -> pd.DataFrame:
    # Parse the XML file
    tree = ET.parse(trackmate_file)
    root = tree.getroot()

    # Find the <Log> element and extract its text content
    log_text = root.find('Log').text

    # Read the TSV data into a pandas DataFrame
    # The StringIO object allows pandas to read the string as if it were a file
    log_data = StringIO(log_text)
    df = pd.read_csv(log_data, sep='\t')
    return df


def find_tip(filepath_tip: str, threshold_tip: int, save_img_to_path: str) -> list:
    img = cv2.imread(filepath_tip, cv2.IMREAD_UNCHANGED)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    tip_mask = blurred_img < threshold_tip
    tip_mask = np.array(tip_mask.astype(int))
    tip_outline = np.array([[0, 0]])
    tip_end = [0, 0]
    img_width, img_height = img.shape[0], img.shape[1]
    print(img_width, img_height)
    for i in range(int(1/4*img_height), int(3/4*img_height)):
        for j in range(0, int(1/4*int(img_width))):
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
            p.image(image=[img], x=0, y=0, dw=img_width, dh=img_height)
            p.circle(x=tip_outline[:, 0], y=tip_outline[:, 1], color='red')
            p.star(x=tip_end[0], y=tip_end[1], color='cyan', size=15)
            if save_img_to_path=='ipynb':
                bokeh.io.show(p)
            else:
                bokeh.io.export_png(p, filename=f"{save_img_to_path}")

        tip_outline = tip_outline[1:]
        return tip_outline, tip_end, tip_mask
 
    else:
        print("No tip found... try another threshold.")
        return False


def filter_tracks(df: pd.DataFrame, tip_mask: np.ndarray, pixel_size: list) -> pd.DataFrame:
    arr_positions = np.divide(df[['POSITION_X', 'POSITION_Y']].values, pixel_size).astype(int)
    filter = tip_mask[arr_positions[:, 1], arr_positions[:, 0]]==0

    df_filtered = df[filter]
    df_filtered = df_filtered[df_filtered['DISTANCE [um]']>25]
    return df_filtered



def add_velocity_to_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add v_x and v_y to the dataframe. 
    '''
    df['vx'] = np.nan
    df['vy'] = np.nan
    df = df.sort_values(by='FRAME')
    particles = df["TRACK_ID"].unique()
    
    for particle in particles:
        track = df.loc[df["TRACK_ID"]==particle]
        if len(track) < 10:
            continue
        
        vx = [np.nan]
        vy = [np.nan]
        for t in range(1, len(track)):
            vx.append((track["POSITION_X"].values[t]-track["POSITION_X"].values[t-1])/(track["POSITION_T"].values[t]-track["POSITION_T"].values[t-1]))

            vy.append((track["POSITION_Y"].values[t]-track["POSITION_Y"].values[t-1])/(track["POSITION_T"].values[t]-track["POSITION_T"].values[t-1]))
            
        df.loc[df["TRACK_ID"]==particle, "vx"] = vx
        df.loc[df["TRACK_ID"]==particle, "vy"] = vy
    return df


def velocity2force(velocity_magnitude: np.ndarray) -> np.ndarray:
    r = 2.8/2 # um
    viscosity = 1412 # mPa.s
    F = 6*np.pi*r*viscosity*velocity_magnitude*0.001       # pN     (M is magnitude of velocities at different positions in um/s)
    return F


def distance_from_tip(tip_outline: np.ndarray, track: pd.DataFrame) -> np.ndarray:
    distances = np.zeros(len(track))
    # print(len(track))
    for j in range(len(track)):
        distance_vectors = np.array(tip_outline)-track[["POSITION_X", "POSITION_Y"]].values[j]
        distances[j] = np.min([np.sqrt(distance_vectors[i][0]**2 + distance_vectors[i][1]**2) for i in range(len(distance_vectors))])
    return distances


def exclude_tracks_inside_tip(tip_outline: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    for i in range(len(df)):
        x, y = df[['POSITION_X', 'POSITION_Y']]
        
        

def velocities(track: pd.DataFrame) -> np.ndarray:
    velocity_vectors = track[["vy", "vx"]].values
    velocities_magnitudes = []
    for i in range(len(velocity_vectors)):
        velocities_magnitudes.append(np.sqrt(velocity_vectors[i][0]**2+velocity_vectors[i][1]**2))
    return np.array(velocities_magnitudes)


def double_exp(x, a1, k1, a2, k2):
    return a1*np.exp(-k1*x)+a2*np.exp(-k2*x)


def single_exp(x, a1, k1):
    return a1*(np.exp(-x/k1))