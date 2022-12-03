# Few-Shot Learning for Music Information Retrieval
## PyTorch Example


This repository contains an example implementation of few-shot learning for musical instrument classification using PyTorch. The code in this repository demonstrates how to train a prototypical network for few-shot learning on a small dataset of musical instrument sounds.

All of the code in this repository is accompanied by a tutorial that can be found [here](https://music-fsl-zsl.github.io/tutorial/fsl-example/intro.html). 

This standalone repository contains self-contained code for training a prototypical network on few-shot learning for musical instrument recogntion, using the [TinySOL](https://zenodo.org/record/3685367) dataset. 


## Installation

To install this repository, clone it and install the requirements:

```bash
git clone https://github.com/music-fsl-zsl/music_fsl
cd music_fsl
pip install -e .
```

## Usage

### Training

To train a prototypical network on the TinySOL dataset, run the following command:

```bash
python -m music_fsl.train 
```

Here are the options that can be passed to the training script:
```bash
Generated arguments for function train:
  The train function trains a few-shot learning model on the
  TinySOL dataset. It takes the following parameters:

  --sample_rate SAMPLE_RATE
                        The sample rate of the audio data.  Default: 16000.
  --n_way N_WAY         The number of classes to sample per episode. Default: 5.
  --n_query N_QUERY     The number of samples per class to use as query. Default:
                        20.
  --n_train_episodes N_TRAIN_EPISODES
                        The number of episodes to generate for training. Default:
                        100000.
  --n_val_episodes N_VAL_EPISODES
                        The number of episodes to generate for validation. Default:
                        100.
  --num_workers NUM_WORKERS
                        The number of worker threads to use for data loading.
                        Default: 10.
```

for more information, run `python -m music_fsl.train -h`.

