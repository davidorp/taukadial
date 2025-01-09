<h1 align="center">Cognitive Insights Across Languages: Enhancing Multimodal Interview Analysis - Taukadial challenge</h3>


<p align="center">
  Cognitive Insights Across Languages: Enhancing Multimodal Interview Analysis
  <br>
  <a href="https://www.isca-archive.org/interspeech_2024/ortizperez24_interspeech.pdf"><img alt="Interspeech" src="https://img.shields.io/badge/Interspeech-2024-blue.svg"></a> 
  <a href="https://arxiv.org/abs/2406.07542"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2406.07542-b31b1b.svg"></a> 
</p>

## Overview

This repository contains the code used for the TAUKADIAL challenge. The work implements a pipeline where various approaches have been considered to tackle the task. The challenge comprises two distinct tasks: predicting Mild Cognitive Impairment (MCI) and forecasting a cognitive score based on the Mini-Mental State Examination (MMSE).

The dataset has been provided by the TalkBank project. Data includes audio recordings in both Chinese and English languages. A Whisper model has been employed to generate transcriptions of the audios. Simultaneously, features from these audios have been extracted using OpenSmile and Opendbm libraries. Bert embeddings have also been utilized from the transcriptions. Various approaches have been proposed to develop a final multimodal model.

To access the Taukadial dataset, please contact the TalkBank project and request permission to use their data.

<div align="center">
  <img src="imgs/overview.png" alt="Overview of proposed architecture">
  <br>
  <em>Figure 1: Overview of proposed architecture</em>
</div>

## Usage

To run our code, the following steps must be performed to obtain the processed data that will fit the deployed models. Please utilize Python version 3.8 to install the opendbm library and acquire the necessary audio features. Once you have obtained and stored the features, you can remove this dependency from the requirements.txt file and utilize a different version of Python.

```sh
sudo apt update
sudo apt install ffmpeg cmake libsndfile1 sox
pip install -r requirements.txt
python3 preprocess_dataset.py
./scripts/train.sh
```

## Citation

```bibtex
@inproceedings{ortizperez24_interspeech,
  title     = {Cognitive Insights Across Languages: Enhancing Multimodal Interview Analysis},
  author    = {David Ortiz-Perez and Jose Garcia-Rodriguez and David Tomás},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {952--956},
  doi       = {10.21437/Interspeech.2024-914},
  issn      = {2958-1796},
}
```
