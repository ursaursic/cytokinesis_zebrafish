#!/bin/bash

# Run the first Python script with its argument
echo "Running 0_add_info_to_df.py with config.yml"
python 0_add_info_to_df.py config.yml

# Run the second Python script with its argument
echo "Running 1_plot_trajectories.py with config.yml"
python 1_plot_trajectories.py config.yml

# Run the third Python script with its argument
echo "Running 2_analyze_material_properties.py with config.yml"
python 2_analyze_material_properties.py config.yml

# Run the fourth Python script with its argument
echo "Running 4_average_curves.py with config.yml"
python 4_average_curves.py config.yml
