# Real-Time Gesture Recognition Using Multi-Channel sEMG Signals with Spatial-Temporal Spiking Neural Network

This repository implements the classification of 6 classes of gestures/actions based on sEMG signals, including:

- **STSNN**: Introduces spiking neurons (SNN) based on the L-EMGNet topology to explore the performance of brain-inspired computing in sEMG classification.

The data consists of multi-channel sEMG + labels, segmented using a sliding window, and supports two types of experimental settings: within-day and cross-day.

## Directory Structure

- `get_data.py`: Reads the original `raw.dat` biosignal files and performs preprocessing and slicing.
  - Reads the custom binary format and parses header information such as sampling rate and the number of channels.
  - Applies a **20–150 Hz 6th-order Butterworth bandpass filter** (zero-phase `filtfilt`) to the first **8 EMG channels**.
  - Slices the data by label using a **500 ms window / 250 ms sliding step**:
    - Sampling rate of 500 Hz → window length of 250 points, step size of 125 points.
    - Only retains segments where all labels within the window are identical.
  - Outputs:
    - `segments.npy`: Data segments with shape `(N, 8, 250)`.
    - `segments_labels.npy`: A label array of length `N`.

- `model.py`: Model definitions.
  - `STSNN`:
    - **Strictly corresponds to the L-EMGNet topology**, only replacing the `ELU` activation with **ParametricLIFNode** (SpikingJelly).
    - Uses `step_mode='m'`, treating the temporal dimension as steps. It unfolds 2D features `(N, C, H, W)` along the temporal dimension into `(T, N, C, H)`, feeds them into the LIF neurons, and then restores them.
    - The goal is to introduce spiking neurons while retaining the representational capacity of EMGNet, striving for performance close to EMGNet.

- `main.py`: Entry point for training and evaluation.
  - Supports **two models**:
    - `--model stsnn`: Uses STSNN (EMGNet + SNN).
  - Supports **two types of experimental settings**:
    - **Within-day experiment**:
      - For Day1 and Day2 of each subject:
        - **First, shuffles** all segments and labels of the corresponding day (using seeded random shuffling to ensure reproducibility).
        - Then splits them into **80% training / 20% testing**.
      - Calculates the test accuracy for Day1 and Day2 separately for each subject, taking the average of the two as the within-day result for that subject.
      - Finally, reports the average within-day accuracy across all subjects.
    - **Cross-day experiment**:
      - For each subject:
        - **Trains on all Day1 data** and **tests on all Day2 data**.
        - Aligns and maps the raw Day2 labels using the Day1 label space as the baseline.
      - Reports the cross-day accuracy for each subject and the average cross-day accuracy across all subjects.
  - Training details:
    - Uses `tqdm` to display epoch-level and batch-level progress bars.
    - Loss function: `CrossEntropyLoss`.
    - Optimizer: `Adam`, with the learning rate configurable via `--lr`.
    - When training on STSNN, calls `reset_net()` before every batch and evaluation to clear the network state.

## Environment Dependencies

A Python 3.8+ environment is recommended. Main dependencies:

- `numpy`
- `scipy` (for Butterworth bandpass filtering)
- `torch` (PyTorch)
- `tqdm`
- `spikingjelly` (only required for STSNN)

Example installation (using pip):


```bash
pip install numpy scipy torch tqdm spikingjelly
```


Data Preparation
--------

The default data directory structure is:

```text
data/
  Sub01/
    Day1/
      raw.dat
    Day2/
      raw.dat
  Sub02/
    Day1/
      raw.dat
    Day2/
      raw.dat
  ...
  Sub10/
    Day1/
    Day2/
```

Each raw.dat is a custom-format biosignal binary file, with the header containing information such as sampling rate and the number of channels.

After running get_data.py, the following will be generated in the corresponding DayX directories:

- `segments.npy`：`(N, 8, 250)`。
- `segments_labels.npy`：`(N,)`。


Running Experiments
--------

1. **Generate Segmented Data**
Execute in the project root directory:

```bash
python get_data.py
```
This will iterate through Sub01–Sub10 and Day1/Day2, performing filtering and slicing on each raw.dat and saving the results.

2. **Single-day + Cross-day Experiments**

```bash
python main.py
```

3. **Run Only Single-day Experiments**

```bash
python main.py --single_day
```

4. **Run Only Cross-day Experiments**

```bash
python main.py --cross_day
```

5. **Model Training and Hyperparameter Examples**

```bash

# Use STSNN (EMGNet topology + LIF) and appropriately reduce Dropout
python main.py --model stsnn --dropout 0.3 --epochs 100 --lr 1e-3
```

License and Citation
----------

- This code can be used for research and educational purposes. If used in papers or projects, please mention this repository in the acknowledgments.

