# Seeing through the conversation: Audio-visual speech separation based on diffusion model (AVDiffuSS)

<img src="https://github.com/cyongong/AVDiffuSS_before/assets/83945744/68a42129-9230-46f6-9258-5c7a8a7140f9" width="800" alt="AVDiffuSS architecture. A predictive model is first used to estimate the clean speech. The following generative stage then uses this estimate as the initial point for a reverse process. For both stages, visual streams are used to extract the target speaker's speech.">


This repository contains the official PyTorch implementation for the paper:

- [Seeing Through the Conversation: Audio-Visual Speech Separation based on Diffusion Model](https://arxiv.org/abs/2310.19581)

Our demo page is [here](https://mmai.io/projects/avdiffuss/).

## Installation

- Create a new virtual environment with the following command. You should change the environment path in yaml file.
- `conda create -n AVDiffuSS python=3.8`
- `pip install -r requirements.txt`

## Pre-trained checkpoints

- We provide pre-trained checkpoint for the model trained for 30 epochs on the VoxCeleb2 train dataset. It can be used for testing on both the VoxCeleb2 and LRS3 test datasets. The file can be downloaded [here](https://drive.google.com/file/d/18nG5cydKwa-yWszqKy_1YpW2mY8QBfre/view?usp=sharing).

Usage:
- For evaluating the pre-trained checkpoint, use the `--testset` option of `test.py` (see section **Evaluation** below) for selecting the test dataset among VoxCeleb2 and LRS3. Use `--ckpt` option to specify the path of the checkpoint for `test.py`.

## Training

For training, run
```bash
python train.py 
```
It you don't want to save checkpoints, add --nolog option.

## Evaluation

To evaluate on a test set, run
```bash
python test.py --testset <'vox' or 'lrs3'> --ckpt /path/to/model/checkpoint --data_dir /path/to/test/data/directory
```
Use 'vox' for VoxCeleb2 test set, and 'lrs3' for LRS3 test set. You can get scores fast since train file only use the first 2.04s per audio for inference. 


If you want to evaluate whole audio, please run
```bash
python test_whole.py --testset <'vox' or 'lrs3'> --ckpt /path/to/model/checkpoint --data_dir /path/to/test/data/directory
```
Inference speed could be faster by changing --hop_length option. Default value is 0.04 which is same with [VisualVoice](https://github.com/facebookresearch/VisualVoice). 


The performance of the provided checkpoint evaluated by the first test command is as follows:

| testset | PESQ | ESTOI | SI-SDR|
|---------|------|-------|-------|
|VoxCeleb2|2.5906|0.8152 |12.2701|
|   LRS3  |2.8106|0.8856 |14.1707|

Since our model is based on a Diffusion method, an inference speech would be slow. That's why we use the first 2.04s audio for checking our scores. 

## Citations

Our paper has been submitted to a conference and is currently under review. Therefore the appropriate citation for our paper may change in the future.

```
@article{lee2023seeing,
  title={Seeing Through the Conversation: Audio-Visual Speech Separation based on Diffusion Model},
  author={Lee, Suyeon and Jung, Chaeyoung and Jang, Youngjoon and Kim, Jaehun and Chung, Joon Son},
  journal={arXiv preprint arXiv:2310.19581},
  year={2023}
}
```