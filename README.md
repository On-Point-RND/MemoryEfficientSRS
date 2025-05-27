## 🎨 SRC_PROJECT
This repository containing implementation Cut Cross Entropy (CCE) and Cut Cross Entropy with Negative Sampling (CCE-) for RecSys. Triton kernels are available in 
`kernels/cut_cross_entropy`. Implementation of SASRec with CCE and CCE- can be found in `src/models/nn/sequential/sasrec/lightning.py`. Experiment pipeline is located in `src_benchmarks`.

## 🚀 Getting Started

Installation via Docker is recommended by default:

```bash
docker build -t src_project .
docker run -it --gpus '"device=0"' src_project
```

### 🐳 Docker Setup for Project
This repository includes a Dockerfile for building a development environment with:
- PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
- Python 3.10 (from the PyTorch base image)
- Java 11 (OpenJDK)

### 📦 Installed System Packages (apt)
The following system dependencies are installed in the image:
- apt-utils – base utility for apt
- build-essential – compiler and toolchain (required by some Python packages)
- libgomp1 – OpenMP support library (used by numpy and other libs)
- pandoc – required for Lightning logs or documentation exports
- git – for cloning repos or versioning
- openjdk-11 – manually copied from the slim OpenJDK image

### 🧪 Python Packages (pip)
Installed Python packages with pinned versions:
```plaintext
  numpy==1.24.4
  lightning==2.5.1
  pandas==1.5.3
  polars==1.0.0
  optuna==3.2.0
  scipy==1.9.3
  psutil==6.0.0
  scikit-learn==1.3.2
  pyarrow==16.0.0
  torch==2.5.1
  rs_datasets==0.5.1
  Ninja==1.11.1.1
  tensorboard==2.19.0
```
## ⚙️ Running Experiments with SASRec
### ⚡️ Quickstart
To run the experiments for training SASRec, use the following command from the project directory:
```bash
python main.py
```
### 📁 Configuration
All experiment parameters are defined using .yaml configuration files located in the `src_benchmarks/configs` directory.
The main configuration file is `src_benchmarks/configs/config.yaml`, where you specify the dataset and model:

```yaml
defaults:
  - dataset: <dataset_name>
  - model: sasrec_<dataset_name>
```

**Available datasets:** 
- movielens_20m
- beauty
- 30music
- zvuk
- megamarket
- gowalla

Each dataset have a corresponding config file, e.g., `src_benchmarks/configs/dataset/movielens_20m.yaml`.

### ⚙️ Custom Loss Configuration
To use CCE- loss, include the following parameters in the model config (sasrec_<dataset_name>.yaml):
```yaml
- loss_type: CCE
- loss_sample_count: <number_of_negative_samples>
```
If `loss_sample_count: null`, the training will use the standard CCE method.

### 🔍 Reproducing CE- Grid Search
We provide a dedicated trainer for grid search, located at:
```bash
src_benchmarks/grid_params_search_runner.py
```
To configure a hyperparameter grid:
1. Modify the file:
`src_benchmarks/configs/mode/hyperparameter_experiment.yaml`
(This controls the grid over `batch_size`, `max_seq_len`, and `loss_sample_count`.)
2. Set the mode to "hyperparameter_experiment" in:
`src_benchmarks/configs/config.yaml`

```yaml
defaults:
  ...
  - mode: hyperparameter_experiment
```

> 💡**P.S.** To adjust other training parameters, edit them in their respective config files (e.g., `src_benchmarks/configs/model/sasrec_movielens_20m.yaml`), not in `hyperparameter_experiment.yaml`.

## 📊 Results

In this section, we present the results of training transformer models **SASRec** and **BERT4Rec** on 6 popular datasets, such as 
- Megamarket
- Zvuk
- 30Music
- Gowalla
- Beauty
- Movielens-20m

The models were trained wwith different loss type function such, as
- Cross Entropy (CE)
- CE with negative sampling (CE-)
- Cut Cross-Entropy (CCE)
- CCE with negative sampling (CCE-)
- Binary cross-entropy with negative sampling (BCE+)
- Scalable CE (SCE)

The results are listed in Tables that include the following details: 
the name of the loss function; 
training hyperparameters such as batch size (BS), maximum sequence length (SL), and the number of negative samples (NN); 
performance metrics (NDCG@10, Coverage@10, and Surprisal@10); memory consumption during training (max_allocated_mem); and the average time spent per epoch (Epoch, s).


Max_allocated_mem refers to the maximum memory consumed for model training, which is calculated using the function `torch.cuda.max_memory_allocated()`.

### Results of experiments on test set for SASRec model

The two tables below present the results for CE and CE-.

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CE     | 16    | 128   |       | 0.005200  | 0.042400  | 0.443800  | 64.7   | 4054.3 |
| Zvuk             | CE     | 64    | 96    |       | 0.079737  | 0.060899  | 0.337537  | 65.2   | 1766.1 |
| 30Music          | CE     | 32    | 128   |       | 0.089100  | 0.049700  | 0.504200  | 69.8   | 252.4  |
| Gowalla          | CE     | 256   | 64    |       | 0.022498  | 0.202788  | 0.632923  | 47.7   | 57.5   |
| Beauty           | CE     | 1024  | 32    |       | 0.011419  | 0.085892  | 0.506053  | 51.7   | 132    |
| Movielens-20m    | CE     | 256   | 512   |       | 0.050973  | 0.241322  | 0.079803  | 19.3   | 49.9   |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CE-    | 1024  | 32    | 255   | 0.011700  | 0.109900  | 0.456000  | 71.0   | 60.0   |
| Zvuk             | CE-    | 1024  | 32    | 256   | 0.067477  | 0.129917  | 0.334126  | 56.0   | 106.0  |
| 30Music          | CE-    | 256   | 128   | 511   | 0.093856  | 0.089839  | 0.524108  | 47.6   | 44.8   |
| Gowalla          | CE-    | 512   | 256   | 512   | 0.027776  | 0.243828  | 0.618486  | 42.7   | 36.7   |
| Beauty           | CE-    | 1024  | 32    | 512   | 0.012330  | 0.121399  | 0.501908  | 11.7   | 36.8   |
| Movielens-20m    |  CE-   | 64    | 512   | 1512  | 0.051288  | 0.192002  | 0.073554  | 39.0   | 233.0  |

Since CE- consumes significantly less memory during training compared to CE, this loss type allows the use of larger BS and SL, resulting in improved metrics. 
Additionally, for datasets with large item catalogs, the training process is considerably accelerated.

The two tables below present the results for CCE and CCE-.
<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CCE    | 5120  | 320   |       | 0.0181    | 0.1705    | 0.5466    | 40.0   | 2248.0   |
| Zvuk             | CCE    | 1024  | 128   |       | 0.0940    | 0.1401    | 0.3547    | 9.5    | 550.0    |
| 30Music          | CCE    | 256   | 720   |       | 0.1393    | 0.1203    | 0.5783    | 9.4    | 389.0    |
| Gowalla          | CCE    | 1024  | 320   |       | 0.0241    | 0.2021    | 0.6447    | 12.0   | 132.0    |
| Beauty           | CCE    | 1024  | 32    |       | 0.0104    | 0.1155    | 0.5172    | 1.5    | 80.3     |
| Movielens-20m    | CCE    | 256   | 512   |       | 0.0511    | 0.2102    | 0.0793    | 3.4    | 43.0     |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CCE-   | 5120  | 320   | 1023  | 0.0168    | 0.1158    | 0.4941    | 80.0     | 262.0    |
| Zvuk             | CCE-   | 1024  | 128   | 2047  | 0.0852    | 0.1490    | 0.3486    | 14.0     | 308.0    |
| 30Music          | CCE-   | 256   | 720   | 4095  | 0.1436    | 0.1241    | 0.5834    | 34.0     | 277.0    |
| Gowalla          | CCE-   | 1024  | 320   | 511   | 0.0279    | 0.2770    | 0.6276    | 12.0     | 21.0     |
| Beauty           | CCE-   | 1024  | 32    | 511   | 0.0116    | 0.1813    | 0.5069    | 2.3      | 23.0     |
| Movielens-20m    | CCE-   | 256   | 512   | 511   | 0.0523    | 0.2429    | 0.0801    | 3.5      | 52.0     | 

CCE and CCE- are much more memory-efficient compared to CE and CE-. As a result, these methods allow the use of larger BS and SL. 
For datasets with large item catalogs, this leads to significant improvements in model performance after training. 

Due to the storage requirements for negative sample indices, CCE- consumes significantly more GPU memory than CCE. 
This contrasts with the behavior observed in CE and CE-, where the memory overhead from materializing logits during the computation of the full CE loss on large item catalogs exceeds the memory used for storing negative sample indices.



The three tables below present the results for BCE, BCE+, gBCE.
<hr>

| Dataset        | Loss type | BS   | SL  | NDCG@10   | Coverage@10 | Surprisal@10 | Max memory, GB | Epoch, s |
|----------------|-----------|------|-----|-----------|-------------|--------------|-------------------|----------|
| Megamarket     | BCE       | 1024 | 32  | 0.004294  | 0.049178    | 0.376738     | 8.9               | 26.9     |
| Megamarket     | BCE       | 5120 | 320 | 0.007127  | 0.055880    | 0.378771     | 36.5              | 55.7     |
| Zvuk           | BCE       | 1024 | 32  | 0.025003  | 0.055162    | 0.299418     | 5.7               | 31.8     |
| Zvuk           | BCE       | 1024 | 128 | 0.030669  | 0.078245    | 0.336043     | 5.7               | 31.8     |
| 30Music        | BCE       | 256  | 128 | 0.080305  | 0.058708    | 0.480891     | 4.9               | 8.9      |
| 30Music        | BCE       | 256  | 720 | 0.091240  | 0.067944    | 0.498069     | 7.4               | 21.6     |
| Gowalla        | BCE       | 512  | 256 | 0.017491  | 0.121836    | 0.569349     | 3.6               | 14.2     |
| Gowalla        | BCE       | 1024 | 320 | 0.019765  | 0.127281    | 0.568624     | 7.2               | 14.3     |
| Beauty         | BCE       | 1024 | 32  | 0.003677  | 0.003602    | 0.452931     | 1.4               | 30.4     |
| Movielens 20m  | BCE       | 64   | 512 | 0.029959  | 0.189254    | 0.087503     | 0.7               | 66.9     |
| Movielens 20m  | BCE       | 256  | 512 | 0.032907  | 0.190266    | 0.087117     | 2.5               | 38.3     |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | BCE+   | 1024  | 32    | 255   | 0.012686  | 0.101091  | 0.480729  | 47.0   | 53.6   |
| Zvuk             | BCE+   | 1024  | 32    | 256   | 0.068039  | 0.118970  | 0.348780  | 24.4   | 48     |
| 30Music          | BCE+   | 256   | 128   | 512   | 0.118101  | 0.106841  | 0.579800  | 47.6   | 42.5   |
| Gowalla          | BCE+   | 512   | 256   | 512   | 0.024939  | 0.294104  | 0.636595  | 42.7   | 36.3   |
| Beauty           | BCE+   | 1024  | 32    | 512   | 0.010525  | 0.091434  | 0.501953  | 11.6   | 36.2   |
| Movielens-20m    | BCE+   | 64    | 512   | 1512  | 0.049955  | 0.195473  | 0.075284  | 39.0   | 238.3  |


Although training with the BCE loss function is more memory- and compute-efficient, the resulting model performance is significantly worse than when using CE-, CCE, or CCE-. 

The table below present the results for SCE.
<hr>

| Dataset        | Loss type | BS   | SL  |  NDCG@10   | Coverage@10 | Surprisal@10 | Max memory, GB | Epoch, s |
|----------------|-----------|------|-----|-----------|-------------|--------------|-------------------|----------|
| Megamarket     | SCE       | 1024 | 32  | 0.0124    | 0.0706      | 0.5278       | 11.3              | 36.4     |
| Megamarket     | SCE       | 2048 | 320 | 0.0160    | 0.0988      | 0.5358       | 38.2              | 144.2    |
| Zvuk           | SCE       | 1024 | 32  | 0.0781    | 0.0804      | 0.3725       | 6.9               | 34.0     |
| Zvuk           | SCE       | 1024 | 128 | 0.0879    | 0.1109      | 0.3711       | 10.5              | 68.4     |
| 30Music        | SCE       | 256  | 128 | 0.1011    | 0.0912      | 0.5682       | 6.4               | 13.7     |
| 30Music        | SCE       | 256  | 720 | 0.1218    | 0.1065      | 0.5812       | 12.2              | 45.5     |
| Gowalla        | SCE       | 512  | 256 | 0.0205    | 0.2805      | 0.6676       | 5.12              | 14.8     |
| Gowalla        | SCE       | 1024 | 320 | 0.0222    | 0.3463      | 0.6887       | 12.4              | 23.6     |
| Beauty         | SCE       | 1024 | 32  | 0.0086    | 0.2290      | 0.5746       | 10.6              | 167.8    |
| Beauty         | SCE       | 1024 | 32  | 0.0088    | 0.2671      | 0.5737       | 10.6              | 167.8    |
| Movielens 20m  | SCE       | 64   | 512 | 0.0477    | 0.1262      | 0.0670       | 6.9               | 34.2     |
| Movielens 20m  | SCE       | 256  | 512 | 0.0396    | 0.1812      | 0.0856       | 6.0               | 69.5     |


Training the model with the SCE loss function yields performance metrics comparable to those of CCE and CCE-, while being more memory-efficient and requiring less time per epoch.
However, since SCE still incorporates the CE loss, it primarily serves as a strategy for item sampling. Its implementation can be further optimized using Triton kernels, enabling more efficient GPU utilization and potentially reducing training time.

It should be noted that in the present study, we focus on optimizing the full CE loss function as well as the CE loss with negative sampling.


### Results for popularity sampling strategy
Similar to CE-, the CCE- method can be applied with various sampling strategies. Below are the results of SASRec training using CCE- with the popularity sampling strategy. 

| Dataset       | Loss type | BS | SL   | NN  |	NDCG@10| Coverage@10| Surprisal@10 |	max_allocated_mem | Epoch, s|
|---------------|-----------|----|------|-----|--------|------------|--------------|--------------|--------|
|Gowalla	      |   CE-     |512 | 256  | 511	|0.018872| 0.476344   | 0.749862     |	42.6       | 105.5 |
|Beauty         |   CE-     |1024| 32   | 511	|0.006125| 0.354296   | 0.700006     |	11.6       | 95.6  |
|Movielens-20m	|   CE-     |64  | 512  | 1511|0.033587| 0.257666   | 0.118191     |	39.0       | 334.8 |

<hr>

| Dataset       | Loss type | BS | SL   | NN  |	NDCG@10| Coverage@10| Surprisal@10 |	max_allocated_mem | Epoch, s|
|---------------|-----------|----|------|-----|--------|------------|--------------|--------------|--------|
|Gowalla	      |   CCE-    |512 | 256  | 511	|0.018959| 0.402872   | 0.747944     |	34.0       | 82.7  |
|Beauty         |   CCE-    |1024| 32   | 511	|0.006619| 0.518114   | 0.709849     |	5.7        | 82.9  |
|Movielens-20m	|   CCE-    |64  | 512  | 1511|0.024230| 0.208128   | 0.124702     |	2.5        | 158.8 |

<hr>

| Dataset       | Loss type | BS | SL   | NN  |	NDCG@10| Coverage@10| Surprisal@10 |	max_allocated_mem | Epoch, s|
|---------------|-----------|----|------|-----|--------|------------|--------------|--------------|--------|
|Gowalla	      |   CCE-    |1024| 320  | 511	|0.020018| 0.454025   | 0.752142     |	67.9       | 86.9  |
|Beauty         |   CCE-    |1024| 32   | 511	|0.006619| 0.518114   | 0.709849     |	5.7        | 82.9  |
|Movielens-20m	|   CCE-    |256 | 512  | 1511|0.023499| 0.222086   | 0.133655     |	6.2        | 111.0 |

It can be observed that, under the popularity sampling strategy, CCE- outperforms CE- on the Gowalla and Beauty datasets.

Additionally, CCE- is more memory-efficient than CE- and requires less time per training epoch.

Compared to the uniform sampling strategy, the popularity sampling strategy results in lower metrics and higher memory consumption during training.
This is attributed to the implementation characteristics of `torch.multinomial`, which can lead to higher memory overhead in popularity-based sampling scenarios.


### Results of experiments on test set for BERT4Rec model

The CCE and CCE- also can be adapted for training BERT4Rec. 

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CE        | 32   | 128 |      | 0.003486  | 0.004417    | 0.360299     | 62.7              | 3684.4  |
| Zvuk             | CE        | 64   | 96  |      | 0.050721  | 0.022123    | 0.264445     | 38.1              | 1514.6  |
| 30Music          | CE        | 32   | 128 |      | 0.023739  | 0.008724    | 0.402552     | 15.4              | 183.2   |
| Gowalla          | CE        | 256  | 64  |      | 0.009060  | 0.095337    | 0.632059     | 12.9              | 30.0    |
| Beauty           | CE        | 1024 | 32  |      | 0.008316  | 0.031619    | 0.487503     | 12.6              | 62.8    |
| Movielens-20m    | CE        | 256  | 512 |      | 0.048726  | 0.188964    | 0.077132     | 7.0               | 32.9    |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CE-       | 1024 | 32  | 255  | 0.003966  | 0.041467    | 0.427789     | 35.4              | 67.1    |
| Zvuk             | CE-       | 1024 | 32  | 256  | 0.000117  | 0.000584    | 0.414634     | 22.8              | 86.4    |
| 30Music          | CE-       | 256  | 128 | 511  | 0.000000  | 0.000021    | 0.613211     | 23.4              | 40.5    |
| Gowalla          | CE-       | 512  | 256 | 512  | 0.000143  | 0.000880    | 0.623571     | 9.9               | 13.2    |
| Beauty           | CE-       | 1024 | 32  | 512  | 0.000045  | 0.000942    | 0.765318     | 2.8               | 21.2    |
| Movielens-20m    | CE-       | 64   | 512 | 1512 | 0.000230  | 0.025673    | 0.192784     | 1.0               | 35.6    |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CCE       | 2048 | 320 |      | 0.012547  | 0.140966    | 0.560321     | 62.8              | 306.3   |
| Zvuk             | CCE       | 1024 | 128 |      | 0.079333  | 0.104216    | 0.357776     | 10.2              | 101.6   |
| 30Music          | CCE       | 256  | 720 |      | 0.088472  | 0.079855    | 0.572631     | 11.6              | 55.9    |
| Gowalla          | CCE       | 1024 | 320 |      | 0.009189  | 0.171003    | 0.654761     | 10.7              | 13      |
| Beauty           | CCE       |1024  | 32  |      | 0.007998  | 0.050229    | 0.509292     | 2.1               | 21      |
| Movielens-20m    | CCE       | 256  | 512 |      | 0.048857  | 0.169004    | 0.074420     | 3.5               | 24.3    |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | CCE-      | 2048 | 320 | 4095 | 0.013025  | 0.135167    | 0.539192     | 53.8              | 206.1   |
| Zvuk             | CCE-      | 1024 | 128 | 2047 | 0.063854  | 0.101359    | 0.343539     | 10.6              | 65.8    |
| 30Music          | CCE-      | 256  | 720 | 4095 | 0.080332  | 0.084903    | 0.560009     | 12.3              | 51.1    |
| Gowalla          | CCE-      | 1024 | 320 | 511  | 0.015484  | 0.155692    | 0.610855     | 10.7              | 10.6    |
| Beauty           | CCE-      | 1024 | 32  | 511  | 0.008504  | 0.030121    | 0.479619     | 2.1               | 21.3    |
| Movielens-20m    | CCE-      | 256  | 512 | 511  | 0.048141  | 0.161701    | 0.072350     | 3.5               | 26.3    |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | BCE+      | 1024 | 32  | 255  | 0.006520  | 0.028739    | 0.393695     | 35.4              | 67.1    |
| Zvuk             | BCE+      | 1024 | 32  | 256  | 0.000126  | 0.001749    | 0.410944     | 22.8              | 86.2    |
| 30Music          | BCE+      | 256  | 128 | 512  | 0.000014  | 0.000021    | 0.698984     | 23.4              | 40.4    |
| Gowalla          | BCE+      | 512  | 256 | 512  | 0.000085  | 0.000331    | 0.770162     | 9.9               | 13.1    |
| Beauty           | BCE+      | 1024 | 32  | 512  | 0.000049  | 0.001484    | 0.731402     | 2.8               | 21.3    |
| Movielens-20m    | BCE+      | 64   | 512 | 1512 | 0.000153  | 0.026323    | 0.197261     | 1.0               | 35.8    |


It can be seen that the memory-efficient CCE and CCE- enable the use of larger batch sizes (BS) and sequence lengths (SL), leading to better metrics for BERT4Rec compared to CE and CE-.

Therefore, CCE and CCE- are useful for both SASRec and BERT4Rec.
