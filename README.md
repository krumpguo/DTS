# FedCoPL
Code of the paper "Stay Unique, Stay Efficient: Preserving Model Personality in Multi-Task Merging"

## Requirements
### Installation
Create a conda environment and install dependencies:
```python
pip install -r requirements.txt
```
### Dataset
We provide implementations on the DTD, RESISC45, UCF101, CUB, CIFAR-10, and CIFAR-100 datasets. To run the experiments, please download the datasets and place them in the dataset directory.

### Usage
Here is an example to run FedCoPL:
```
nohup bash run-textPT_ALL.sh
```
### Acknowledgement
We borrow some code from [CPL](https://github.com/vanillaer/CPL-ICML2024)
