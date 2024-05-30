# FOR GENERATTING THE DATASET
## Installation

```bash
pip install blender numpy pandas
```

## Usage
Generate the dataset by running the `blend.py` file in the blender folder. 
The dataset will be saved in the datasets folder.

```bash
python blender/blend.py
```

To calculate angle to the closest matrices in training set for each test matrix run with appropriate modifications in the code.

```bash
python blender/closest.py
```

# FOR TRAINING THE NETWORK 
## Installation

```bash
# setup conda
conda create -n bcenv python=3.12
conda activate bcenv

# install CUDA enabled PyTorch (https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# install other dependencies
conda install -c conda-forge numpy matplotlib opencv psutil scipy kornia prettytable pandas
```

# GENERATED DATASET
## Usage
First genereta the folders for trianing data by running (with appropriate modifications in the code):
```bash 
python siet/create_folders.py
```

To train the network use `train.py` file with appropriate modifications in the code.

```bash
python siet/my_train.py
```

To infer the results on the test data use `infer.py` file with appropriate modifications in the code.

```bash
python siet/my_infer.py
```

To evaluate and save the results on the test data use `eval.py` file with appropriate modifications in the code.

```bash
python siet/my_eval.py
```

    
# BINS

## Acquiring the dataset
Download the dataset into folder named `bins` from 
[this link](https://liveuniba-my.sharepoint.com/:f:/g/personal/madaras2_uniba_sk/ElUx1HrcIWZPnLT3y_uLYQ0B7k-xgrzu_3Matt7CgDfVTg?e=xO0CTJ).

Note that the dataset has 100GB and the download may fail at around 20GB mark. In that case, download the files in smaller chunks.


## Usage
To train the network use `train.py` file with appropriate modifications in the code.

```bash
python siet_for_bins/train.py
```

To infer the results on the test data use `infer.py` file with appropriate modifications in the code.

```bash
python siet_for_bins/infer.py
```
To evaluate and save the results on the test data use `eval.py` file with appropriate modifications in the code.

```bash
python siet_for_bins/eval.py
```
