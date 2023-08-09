**PyTorch implementation of the paper 
"_<ins>**D**</ins>eep <ins>**M**</ins>omentum  Multi-Marginal <ins>**S**</ins>chrödinger <ins>**B**</ins>ridge_  (**DMSB**)"**

**For NeurIPS 2023 Rebuttal Only. The code is not finally cleaned up, but the results are reproduciable by the following command lines.**


# Installation

This code is developed with Python3. PyTorch >=1.7 (we recommend 1.8.1). First, install the dependencies with [Anaconda](https://www.anaconda.com/products/individual) and activate the environment `DMSB` with
```bash
conda env create --file requirements.yaml python=3.8
conda activate DMSB
```

# Reproducing the result in the paper
****

### GMM
```bash
python main.py --problem-name gmm --dir NeurIPS/gmm --log-tb --use-amp
```
**Memo: The results in the paper sould be reproduced by around 6 stage of Bregman Iteration.**

### Petal
```bash
python main.py --problem-name petal --dir NeurIPS/petal --log-tb --num-stage 30 --use-amp --seed 99
```
**Memo: The results in the paper sould be reproduced by around 17 stage of Bregman Iteration.**

### RNAsc
```bash
python main.py --problem-name RNAsc --dir NeurIPS/RNA-seed99 --log-tb --seed 99  --use-amp --num-itr 2000
```
**Memo: The results in the paper sould be reproduced by around 2-3 stage of Bregman Iteration. (Approximately 44mins on one RTX 3090 Ti as been reported in the rebuttal.)**

**However, the results sould be better than all the baselines in the 1st Bregman Iteration.**

****
# Where Can I find the results?
The visualization results are saved in the folder `/results`.

The numerical value are saved in the tensorboard and event file are saved the folder `/runs`. 

The numerical results for all metrics will be displayed in the terminal as well.