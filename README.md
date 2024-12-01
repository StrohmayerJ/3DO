# PyTorch Dataloader for the 3DO Dataset

### Paper
**Strohmayer J., and Kampel M.** (2024), “On the Generalization of WiFi-based Person-centric Sensing in Through-Wall Scenarios”, International Conference on Pattern Recognition (ICPR), December 2024, Kolkata, India. doi: https://doi.org/10.1007/978-3-031-78354-8_13

BibTeX:
```
@inproceedings{strohmayerOn2024,
  title={On the Generalization of WiFi-based Person-centric Sensing in Through-Wall Scenarios},
  author={Strohmayer, Julian and Kampel, Martin},
  booktitle={International Conference on Pattern Recognition},
  pages={194-211},
  year={2024},
  organization={Springer}
}
```
### WiFi System
<img src="https://github.com/user-attachments/assets/79caebc8-6d96-4726-a88f-dfee70093980" alt="System" width="300"/>

**Strohmayer, J., and Kampel, M.** (2024). “WiFi CSI-based Long-Range Person Localization Using Directional Antennas”, The Second Tiny Papers Track at ICLR 2024, May 2024, Vienna, Austria. https://openreview.net/forum?id=AOJFcEh5Eb

BibTeX:
```
@inproceedings{
strohmayer2024wifi,
title={WiFi {CSI}-based Long-Range Person Localization Using Directional Antennas},
author={Julian Strohmayer and Martin Kampel},
booktitle={The Second Tiny Papers Track at ICLR 2024},
year={2024},
url={https://openreview.net/forum?id=AOJFcEh5Eb}
}
```


### Prerequisites
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset
Get the 3DO dataset from https://zenodo.org/records/10925351 and put it in the `/data` directory.

### Training & Testing 
Example command for training & testing a dummy ResNet18 model on CSI amplitude features with a window size of 351 WiFi packets:

```
python3 train.py --data /data/3DO --bs 128 --ws 351 
```
In this configuration, the samples will have a shape of [128,1,52,351] = [batch size, channels, subcarriers, window size].

