'''
Ursa, updated: 22.1.2024
This script is used to plot calculated effective viscosities from tracks, which were analysed with "calculate_viscosity_from_tracks.py" file.

use pol_stats environment: conda activate pol_stats
'''

import pandas as pd
import numpy as np

import bokeh.io
import bokeh.plotting
import bokeh.models
import iqplot

# Do you want to use data with or without substracted background flows?
substract_background = False

# Define directory to save plots in
plots_dir = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers/3_plots/'

def main():
    if substract_background:
        results_dir = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers/2_analysis/results_viscosity_corrected.csv'
    else:
        results_dir = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers/2_analysis/results_viscosity.csv'

    df_all_results = pd.read_csv(results_dir)

    # Plot results for separate files
    for (idx, filename) in enumerate(df_all_results['FILENAME'].unique()):
        data = df_all_results[df_all_results['FILENAME']==filename]
        p = iqplot.stripbox(data[data['PHASE'].notna()], spread='jitter', q='VISCOSITY', cats='PHASE')
        p.xaxis.axis_label = 'Viscosity (mPas)'
        p.title = f'{filename}'
        if substract_background:
            bokeh.io.output_file(filename=f"{plots_dir}effective_viscosities/viscosity_corrected_{filename}.html", title=f'{filename}_corrected')
            bokeh.io.save(p)
        else:
            bokeh.io.output_file(filename=f"{plots_dir}effective_viscosities/viscosity_{filename}.html", title=f'{filename}')
            bokeh.io.save(p)

    # Plot results for all the files together
    p = iqplot.stripbox(df_all_results[df_all_results['PHASE'].notna()], spread='jitter', q='VISCOSITY', cats='PHASE')
    p.xaxis.axis_label = 'Viscosity (mPas)'

    if substract_background:
        bokeh.io.output_file(filename=f"{plots_dir}effective_viscosities/viscosity_all_samples_corrected.html", title='viscosity_all_samples_corrected')
        bokeh.io.save(p)
    else:
        bokeh.io.output_file(filename=f"{plots_dir}effective_viscosities/viscosity_all_samples.html", title='viscosity_all_samples')
        bokeh.io.save(p)


if __name__ == "__main__":
    main()
    print("Done! :)")