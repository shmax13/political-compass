# Political Compass

A toolkit of classification and regression methods for analyzing political speeches, created by **Alma Linder** and **Maximilian Pfeil** for the **NLP** course at Reykjavik University (during WS24).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

---

## Installation

1. **Clone the repo**:  
   `git clone https://github.com/shmax13/political-compass.git`

2. **Install dependencies**:  
   `poetry install --no-root`

## Usage

1. **Download Speech Corpus**:  
   `python3 util/download_mc_speeches.py`

2. **Run Preprocessing**:  
   `python3 util/preprocessing.py`

3. **Train and Evaluate Models**:  
   `python3 main.py`

4. **Run Gradio UI**:  
   `python3 gradio_ui.py`

## Components & Evaluation

The folders `classifiers`, `regressors` and `vectorizers` contain models pre-trained by us, so the Gradio UI can be run without having to train the models yourself first.  
A `main` run takes about 10 minutes on an Apple Silicon M2 chip. `preprocessing`takes 2-3 minutes.  
A log of evaluation results is found in `eval.txt`. A more detailed discussion is done in `report.pdf`. 