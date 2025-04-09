# Neural-Spike-Detection-and-Analysis-using-PCA
# Description
This repository contains MATLAB scripts developed for the detection, sorting, and analysis of neuronal spike data. It utilizes signal processing techniques and Principal Component Analysis (PCA) to differentiate action potentials (spikes) from noise.
## Features
- Data Import and Preprocessing: Load and preprocess neural signal data sampled at 30 kHz.
- Threshold Determination: Calculate thresholds based on signal variance to identify action potentials.
- Spike Detection: Extract action potential snippets around threshold crossings.
- Noise Generation: Develop methods to systematically sample noise snippets from neural recordings.
- Principal Component Analysis: Custom PCA implementation to distinguish signal from noise.
- Visualization: Plot raw neural signals, threshold levels, spike waveforms, and PCA clustering results in both 2D and 3D visualizations.

## Structure
- data/ - Contains example neural data files (.mat).
- scripts/ - MATLAB scripts performing analysis.
- figures/ - Generated plots for spikes and PCA clustering.

## How to Run
- Clone this repository:
  git clone https://github.com/yourusername/neural-spike-analysis.git
- Open MATLAB and navigate to the repository directory.
- Run the main script:
main_spike_analysis.m

## Dependencies
- MATLAB
- Signal Processing Toolbox (MATLAB built-in functions)

## Contribution
Feel free to fork and contribute to this project by creating pull requests or reporting issues.
