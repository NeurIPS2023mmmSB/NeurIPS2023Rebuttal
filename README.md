**PyTorch implementation of the paper 
"_<ins>**D**</ins>eep <ins>**M**</ins>omentum  Multi-Marginal <ins>**S**</ins>chrödinger <ins>**B**</ins>ridge_  (**DMSB**)"**

**For NeurIPS 2023 Rebuttal Only. The code is not finally cleaned up, but the results are reproduciable by the following command lines.**


# Installation

This code is developed with Python3. PyTorch >=1.7 (we recommend 1.8.1). First, install the dependencies with [Anaconda](https://www.anaconda.com/products/individual) and activate the environment `DMSB` with
```bash
conda env create --file requirements.yaml python=3.8
conda activate DMSB
```

## RNAsc data Preparation
Download the RNAsc data from [here](https://github.com/KrishnaswamyLab/TrajectoryNet/blob/master/data/eb_velocity_v5.npz), and create a folder `data/` and put the downloaded dataset `eb_velocity_v5.npz` into `data/`. 
# Reproducing the result in the paper
****
We provide the checkpoint and the code for training from scratch for all the dataset reported in the paper.

### GMM
```bash
python main.py --problem-name gmm --forward-net toy --backward-net toy --log-tb --samp-bs 2000   --ckpt-freq 10 --num-itr 1000  --num-stage 15 --T 1  --train-bs-x 256 --num-epoch 1  --sigma-max 4  --dir reproduce/gmm   --v-sampling langevin --use-corrector --snr 0.1  --use-amp  --T 3 --interval 300  --var 0.5 --v-scale 3 --reg 0.5
```
**Memo: The results in the paper sould be reproduced by around 6 stage of Bregman Iteration.**

### Petal
```bash
python main.py --problem-name gmm --forward-net toy --backward-net toy --log-tb --samp-bs 2000   --ckpt-freq 10 --num-itr 1000  --num-stage 15 --T 1  --train-bs-x 256 --num-epoch 1  --sigma-max 4  --dir reproduce/gmm   --v-sampling langevin --use-corrector --snr 0.1  --use-amp  --T 3 --interval 300  --var 0.5 --v-scale 3 --reg 0.5
```
**Memo: The results in the paper sould be reproduced by around 17 stage of Bregman Iteration.**

### RNAsc
```bash
python main.py --problem-name gmm --forward-net toy --backward-net toy --log-tb --samp-bs 2000   --ckpt-freq 10 --num-itr 1000  --num-stage 15 --T 1  --train-bs-x 256 --num-epoch 1  --sigma-max 4  --dir reproduce/gmm   --v-sampling langevin --use-corrector --snr 0.1  --use-amp  --T 3 --interval 300  --var 0.5 --v-scale 3 --reg 0.5
```
**Memo: The results in the paper sould be reproduced by around 2-3 stage of Bregman Iteration. (Approximately 44mins on one RTX 3090 Ti as been reported in the rebuttal.)**

**However, the results sould be better than all the baselines in the 1st Bregman Iteration.**

****
# Where Can I find the results?
The visualization results are saved in the folder `/results`.
The numerical value are saved in the tensorboard and event file are saved the folder `/runs`. The numerical results for all metrics will be displayed in the terminal as well.