#!/bin/bash
cd ~/PhD/Projects/1_DataAnalysis_zebrafish/beads_in_zebrafish/magnetic_tweezers_SD
conda activate cytokinesis_zeb

# Run the first Python script with its argument
echo "Running 0_add_info_to_df.py with config_noco.yml"
python 0_add_info_to_df.py config_noco.yml

# Run the second Python script with its argument
echo "Running 1_plot_trajectories.py with config_noco.yml"
python 1_plot_trajectories.py config_noco.yml

# Run the third Python script with its argument
echo "Running 2_analyze_material_properties.py with config_noco.yml"
python 2_analyze_material_properties.py config_noco.yml

# Run the fourth Python script with its argument
echo "Running 3_average_curves.py with config.yml"
python 3_average_curves.py config_noco.yml

# Run the fifth  Python script with its argument
echo "Running 4_statistical_analysis.py with config.yml"
python 4_statistical_analysis.py config_noco.yml
