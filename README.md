# Analysis of Various Self-Supervised Learning Models for Automatic Pronunciation Assessment (APSIPA 2024)

This repository contains the official codebase for the paper:

> **"Analysis of Various Self-Supervised Learning Models for Automatic Pronunciation Assessment"**  
> *Haeyoung Lee, Sunhee Kim, Minhwa Chung*  
> Accepted at **APSIPA ASC 2024 (Asia-Pacific Signal and Information Processing Association Annual Summit and Conference)**

📄 [**Read the Paper**]([https://ieeexplore.ieee.org/abstract/document/10848954)
📌 DOI: [10.1109/APSIPAASC63619.2025.10848954](https://doi.org/10.1109/APSIPAASC63619.2025.10848954)

---

## 🧠 Overview

This repository provides code and experimental setups for analyzing and fine-tuning **Self-Supervised Learning (SSL)** speech models for **Automatic Pronunciation Assessment (APA)**.  
We evaluate 12 pretrained SSL models (Wav2Vec2.0, HuBERT, WavLM) under three fine-tuning strategies:

- **CTC Head**  
- **Freezing CNN Feature Extractor**  
- **No CTC / General Feature Extraction**

The study further introduces a novel **PCA-based intrinsic analysis** method that interprets model behavior by analyzing feature manifolds.

---

## 📊 Key Contributions

- First **systematic benchmark** of SSL models for APA
- Comparison of 12 SSL variants using Speechocean762 dataset
- Dual **extrinsic and intrinsic analysis** of performance and scoring behavior
- PCA-based visualization of hidden representation structures
- Identification of optimal SSL models for different APA goals

---

## 📁 Repository Structure

```
ssl_finetuning/
├── train/                   # Fine-tuning scripts for each model & setting
├── test/                    # Evaluation scripts
├── analysis/                # PCA, score correlation, and visualization
├── calculate_pcc/           # PCC computation utilities
├── requirements.txt         # Dependency list
└── *.ipynb                  # Jupyter notebooks used for figure generation
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/hy310/ssl_finetuning.git
cd ssl_finetuning
pip install -r requirements.txt
```

### 2. Dataset

This repository uses the [**Speechocean762**](https://www.openslr.org/103/) corpus.  
Make sure to preprocess and store it in the following structure:

```
/your/data/path/speechocean762/
├── preprocess/
│   ├── speechocean_train_ds/
│   └── speechocean_test_ds/
```

### 3. Fine-tuning

To fine-tune a model (e.g., `hubert-xlarge` with CTC):

```bash
python train/train_ctc.py \
  --model_name facebook/hubert-xlarge-ls960-ft \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 30
```

### 4. Evaluation

```bash
python test/evaluate_model.py \
  --model_path /path/to/saved/model.pt
```

---

## 📈 Example Results (Speechocean762)

| Model                   | Accuracy | Fluency | Prosody | Total |
|------------------------|----------|---------|---------|-------|
| `wav2vec2-large`       | 0.691    | **0.794** | 0.786   | 0.728 |
| `hubert-xlarge-ft`     | **0.722** | **0.797** | **0.788** | **0.734** |
| `wavlm-large`          | 0.656    | 0.736   | 0.726   | 0.680 |

---

## 📊 PCA-Based Intrinsic Analysis

We propose a novel intrinsic interpretability method based on **PCA of hidden representations**.

<img src="https://raw.githubusercontent.com/hy310/ssl_finetuning/main/analysis/figure/pca_3d_example.png" width="400"/>

- **Conical (Wav2Vec2.0)**: emphasizes score continuity
- **V-shape (HuBERT)**: two-axis decision
- **S-shape (WavLM)**: diverse scoring factors

---

## 📎 Citation

If you use this repository or our findings, please cite:

```bibtex
@inproceedings{lee2024sslapa,
  title     = {Analysis of Various Self-Supervised Learning Models for Automatic Pronunciation Assessment},
  author    = {Haeyoung Lee and Sunhee Kim and Minhwa Chung},
  booktitle = {APSIPA Annual Summit and Conference (APSIPA ASC)},
  year      = {2024},
  doi       = {10.1109/APSIPAASC63619.2025.10848954}
}
```

---

## 🙌 Acknowledgements

This project was conducted at **Seoul National University**,  
within the **Interdisciplinary Program in Cognitive Science**  
and supported by the **SNU Spoken Language Processing Lab**.

---

## 📬 Contact

For questions or collaborations, please contact:

- **Haeyoung Lee** – `haeylee@snu.ac.kr`
