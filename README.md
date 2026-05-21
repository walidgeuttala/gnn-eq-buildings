# GNN Earthquake Buildings: Structural Seismic Response Prediction Framework

## Project Overview

This repository contains the code for a novel deep learning framework designed to rapidly and accurately predict structural seismic responses. Conventional data-driven methods for predicting seismic response of structures often require extensive data and computational resources. 

To overcome these limitations, this framework utilizes **Transfer Learning** based on the most relevant knowledge determined via **Unsupervised Learning** techniques. 

### How it Works
1. **Unsupervised Learning (Finding the Match):** The framework leverages a seismic information history database to identify the most similar previous earthquake.
2. **Transfer Learning (Knowledge Transfer):** It transfers the corresponding knowledge from a Structural Seismic Response network (SSR net) to predict structural responses caused by a new earthquake.

This approach significantly reduces the need for extensive data collection and provides highly efficient predictions. It reliably captures the complex nonlinear dynamics of structures under seismic loads and offers significant potential for advancing seismic fragility analyses and reliability assessments.

## Repository Structure

The primary source code is located within the `PRJ-4014v3/` directory:

- **`PRJ-4014v3/TransferLSTM.py`**: The core script containing the LSTM (Long Short-Term Memory) model and Transfer Learning logic to predict displacement/responses.
- **`PRJ-4014v3/find_1nn.py`**: Unsupervised learning component for identifying the most similar prior earthquake (1-Nearest Neighbor).
- **`PRJ-4014v3/dataprocessing.py`**: Data parsing and sequence generation for training the models.
- **`PRJ-4014v3/hyper-tuning.py`**: Hyperparameter tuning script for optimizing the model architecture and training process.

### Case Studies & Data
The repository is applied to multiple structural case studies. Each folder contains source data and targets:
- `PRJ-4014v3/High-rise building/`
- `PRJ-4014v3/Highway bridge/` (e.g., Ferndale, Petrolia)
- `PRJ-4014v3/Six-story building/` (e.g., Chinohills, Devore)

## Prerequisites

The project requires Python and the following key libraries:
- `tensorflow` / `keras`
- `pandas`
- `numpy`
- `scikit-learn`
- `hyperopt` (for hyperparameter tuning)
- `matplotlib`

## Usage

1. **Data Preparation:** Use `dataprocessing.py` to structure your structural response and ground motion (GM) datasets.
2. **Finding the Nearest Earthquake:** Use `find_1nn.py` to query your historical database for the most relevant prior event.
3. **Training & Transfer Learning:** Configure `TransferLSTM.py` to point to the relevant datasets inside the structural folders. Execute the script to train the model and output time-force predictions and charts.
4. **Hyperparameter Tuning:** Use `hyper-tuning.py` to experiment with network depths, sequence windows (`n_lookback`, `n_forecast`), and unit counts.

## License & Credits
**Title:** A Real-time Structural Seismic Response Prediction Framework based on Transfer Learning and Unsupervised Learning
**Authors:** Hongrak Pak, Stephanie Paal (Texas A&M University)
**Funding:** NSF
**Original Dataset/Code Publication DOI:** 10.17603/ds2-fx45-dd16 (DesignSafe)
