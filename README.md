# Dual-Ensemble-System-with-Treble-Former

## Quick start

### 1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

### 2. [Install PyTorch](https://pytorch.org/get-started/locally/)

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Preparing datasets and pre-training models
(1) If training, validation and testing are done on the same dataset (Kvasir or CVC-ClinicDB is recommended for this 
dataset), put the dataset into "data1", and train_with_data1.py will automatically split the dataset into training, 
validation and testing according to 8:1:1.\
(2) If training, validation and testing are not in the same dataset or in the same center, put the dataset for training
and validation into "data2/train_and_val" and the dataset for testing into "data2/test", and train_with_data2.py will 
split the dataset for training and validation by itself according to 9:1:1.\
(3) The datasets used in this study are publicly available at: \
Kvasir-SEG: https://datasets.simula.no/kvasir-seg/. \
CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/. \
ETIS-LaribpolypDB: https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=share_link. \
CVC-ColonDB: https://drive.google.com/drive/folders/1-gZUo1dgsdcWxSdXV9OAPmtGEbwZMfDY?usp=share_link. \
PolypGen: https://www.synapse.org/#!Synapse:syn45200214.

### 5. run training:
```
python train_with_data1.py --amp -Ename t16K -e 100 -b 4 -n1 TrebleFormer_L -n2 TrebleFormer_S -n3 FCBFormer_L -n4 FCBFormer_S -n5 ESFPNet_L -n6 ESFPNet_S -nN 6
python train_with_data2.py --amp -Ename t16K -e 100 -b 4 -n1 TrebleFormer_L -n2 TrebleFormer_S -n3 FCBFormer_L -n4 FCBFormer_S -n5 ESFPNet_L -n6 ESFPNet_S -nN 6
```
#### If you use, please cite:
[Xu, C. et al. Dual Ensemble System for Polyp Segmentation with Multi-Head Control Ensemble and Sub-Model Adaptive Selection Ensemble]
