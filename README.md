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

Max_allocated_mem is the memory consumption value calculated using the function ```torch.max_allocated_memory()``` after training the model.

### Results of experiments on test set for BERT4Rec model:

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

### Results of experiments on test set for SASRec model:

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

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | BCE    | 16    | 128   |       | 0.001741  | 0.00336   | 0.277708  | 78.6   | 10684.0|
| Zvuk             | BCE    | 64    | 96    |       | 0.004551  | 0.000941  | 0.123522  | 60.6   | 2191.0 |
| 30Music          | BCE    | 32    | 128   |       | 0.000354  | 0.000029  | 0.310991  | 4.4    | 35.0   |
| Gowalla          | BCE    | 256   | 64    |       | 0.019690  | 0.246198  | 0.618782  | 45.5   | 83.8   |
| Beauty           | BCE    | 1024  | 32    |       | 0.009411  | 0.060450  | 0.488240  | 21.2   | 107.3  |
| Movielens-20m    | BCE    | 256   | 512   |       | 0.055727  | 0.200969  | 0.075262  | 8.7    | 42.9   |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | BCE+   | 1024  | 32    | 255   | 0.012686  | 0.101091  | 0.480729  | 47.0   | 53.6   |
| Zvuk             | BCE+   | 1024  | 32    | 256   | 0.068039  | 0.118970  | 0.348780  | 24.4   | 48     |
| 30Music          | BCE+   | 256   | 128   | 512   | 0.118101  | 0.106841  | 0.579800  | 47.6   | 42.5   |
| Gowalla          | BCE+   | 512   | 256   | 512   | 0.024939  | 0.294104  | 0.636595  | 42.7   | 36.3   |
| Beauty           | BCE+   | 1024  | 32    | 512   | 0.010525  | 0.091434  | 0.501953  | 11.6   | 36.2   |
| Movielens-20m    | BCE+   | 64    | 512   | 1512  | 0.049955  | 0.195473  | 0.075284  | 39.0   | 238.3  |

<hr>

| Dataset          | Loss type | BS   | SL  | NN   | NDCG@10   | Coverage@10 | Surprisal@10 | max_allocated_mem | Epoch, s|
|------------------|-----------|------|-----|------|-----------|-------------|--------------|-------------------|---------|
| Megamarket       | gBCE   | 1024  | 32    | 256   | 0.006172  | 0.009619  | 0.374430  | 26.3   | 47.2   |
| Zvuk             | gBCE   | 1024  | 32    | 256   | 0.049307  | 0.030365  | 0.307342  | 24.4   | 47.0   |
| 30Music          | gBCE   | 256   | 128   | 512   | 0.000695  | 0.000057  | 0.272706  | 47.6   | 42.9   |
| Gowalla          | gBCE   | 512   | 256   | 512   | 0.018421  | 0.202327  | 0.652298  | 42.7   | 36.3   |
| Beauty           | gBCE   | 1024  | 32    | 512   | 0.009323  | 0.094808  | 0.520879  | 11.6   | 36.4   |
| Movielens-20m    | gBCE   | 64    | 512   | 1512  | 0.049496  | 0.210298  | 0.079313  | 39.1   | 237.4  |

### Results for popularity sampling strategy

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