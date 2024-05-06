# Installation

```bash
# setup conda
conda create -n bcenv python=3.12
conda activate bcenv

# install CUDA enabled PyTorch (https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# install other dependencies
conda install -c conda-forge numpy matplotlib opencv psutil scipy
```

# Tmux
```bash
vim ~/.tmux.conf
set -g mode-mouse on      ###Insert this setting with vim, then source the file.
tmux source-file ~/.tmux.conf
```