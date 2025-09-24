# Analysis of Various Self-Supervised Learning Models for Automatic Pronunciation Assessment (APSIPA ASC 2024)

This repository contains the official codebase for the paper:

> **"Analysis of Various Self-Supervised Learning Models for Automatic Pronunciation Assessment"**  
> *Haeyoung Lee, Sunhee Kim, Minhwa Chung*  
> Accepted at **APSIPA ASC 2024 (Asia-Pacific Signal and Information Processing Association Annual Summit and Conference)**

📄 [**Read the Paper**](https://ieeexplore.ieee.org/abstract/document/10848954)

---

## 🧠 Overview

This repository provides code and experimental setups for analyzing and fine-tuning **Self-Supervised Learning (SSL)** speech models for **Automatic Pronunciation Assessment (APA)**.  
We evaluate 12 pretrained SSL models (Wav2Vec2.0, HuBERT, WavLM) under three fine-tuning strategies:

- **with CTC Head**  
- **Freezing CNN Feature Extractor**  
- **No CTC / General Feature Extraction**

The study further introduces a novel **PCA-based intrinsic analysis** method that interprets model behavior by analyzing feature manifolds.

---

## 📊 Key Contributions

- First **systematic analysis** of SSL models for APA
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
└── requirements.txt         # Dependency list
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
This repository uses the [**Speechocean762**](https://openslr.org/101/) corpus.  
We **preprocessed the dataset using `preprocess_dataset.py`**.  

Make sure to preprocess and store it in the following structure:

```
/your/data/path/speechocean762/
├── preprocess/
│   ├── speechocean_train_ds/
│   └── speechocean_test_ds/
```

### 3. Fine-tuning

To fine-tune a model (e.g., `hubert-xlarge` without CTC):

```bash
python train/baseline.py \
  --model_name facebook/hubert-xlarge-ls960-ft \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 30
```

To fine-tune the same model **with frozen CNN feature extractor** (Freeze mode):

```bash
python train/freeze.py \
  --model_name facebook/hubert-xlarge-ls960-ft \
  --freeze_feature_extractor \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 30
```

## 🤗 Fine-tuned Models

The fine-tuned models introduced in our paper are available on the Hugging Face Hub:
  
- **Model Hub**: https://huggingface.co/haeylee/ssl_ft_pron

> The repository contains multiple subdirectories (e.g., `wav2vec2/general/01_wav2vec2-large`, `wav2vec2/general/02_wav2vec2-large-960h`, …). Pick the subdirectory corresponding to the variant you want to load.

You can directly load a model checkpoint with the 🤗 Transformers library. For example:

### A) CTC Models (with CTC head)
```python
from transformers import AutoModelForCTC, AutoProcessor
model = AutoModelForCTC.from_pretrained("haeylee/ssl_ft_pron/wav2vec2/ctc/01_wav2vec2-large")
processor = AutoProcessor.from_pretrained("haeylee/ssl_ft_pron/wav2vec2/ctc/01_wav2vec2-large")
```
### B) General / Freeze Models (No CTC head)
```python
from transformers import Wav2Vec2Model, HubertModel, WavLMModel
model = Wav2Vec2Model.from_pretrained("haeylee/ssl_ft_pron/wav2vec2/general/01_wav2vec2-large")
processor = AutoProcessor.from_pretrained("haeylee/ssl_ft_pron/wav2vec2/general/01_wav2vec2-large")
# or:
# model = HubertModel.from_pretrained("haeylee/ssl_ft_pron/hubert/freeze/06_hubert-large-ll60k")
# processor = AutoProcessor.from_pretrained("haeylee/ssl_ft_pron/hubert/freeze/06_hubert-large-ll60k")
```


## 📈 Example Results (Speechocean762)

The table below reproduces **Table I** from our paper and presents the **Pearson Correlation Coefficient (PCC)** between model-predicted scores and human annotations across four aspects of pronunciation: **Accuracy**, **Fluency**, **Prosody**, and **Total**.

- **Bold** values indicate the best model overall for each metric.
> Higher PCC values indicate stronger correlation between predicted and true pronunciation scores, reflecting better assessment performance.

<small><small>
| Model               | Acc (No CTC) | Acc (CTC) | Acc (Freeze) | Flu (No CTC) | Flu (CTC) | Flu (Freeze) | Pros (No CTC) | Pros (CTC) | Pros (Freeze) | Total (No CTC) | Total (CTC) | Total (Freeze) |
|--------------------|---------------|-----------|--------------|--------------|-----------|--------------|---------------|------------|---------------|----------------|-------------|----------------|
| w2v2-large         | 0.691         | 0.688     | 0.694        | **0.794**    | 0.787     | 0.782        | **0.786**     | 0.785      | 0.776         | 0.728          | 0.718       | 0.723          |
| w2v2-large-960h    | **0.706**     | 0.708     | 0.702        | 0.773        | 0.770     | 0.774        | 0.773         | 0.771      | 0.775         | **0.734**      | 0.729       | 0.727          |
| w2v2-large-lv60    | 0.623         | 0.666     | 0.649        | 0.676        | 0.720     | 0.749        | 0.672         | 0.730      | 0.742         | 0.642          | 0.686       | 0.679          |
| w2v2-xlsr-53       | 0.678         | 0.691     | 0.645        | 0.740        | 0.752     | 0.694        | 0.734         | 0.751      | 0.691         | 0.694          | 0.706       | 0.664          |
| w2v2-xls-r-300m    | 0.633         | 0.649     | 0.661        | 0.693        | 0.705     | 0.735        | 0.681         | 0.692      | 0.727         | 0.647          | 0.663       | 0.679          |
| hb-large-ll60k     | 0.620         | 0.616     | 0.698        | 0.692        | 0.687     | 0.763        | 0.683         | 0.681      | 0.760         | 0.633          | 0.633       | 0.716          |
| hb-base-ls960      | 0.673         | 0.626     | 0.674        | 0.760        | 0.708     | 0.743        | 0.759         | 0.693      | 0.739         | 0.704          | 0.649       | 0.698          |
| hb-xlarge-ll60k    | 0.631         | 0.686     | 0.702        | 0.704        | 0.759     | 0.786        | 0.693         | 0.761      | 0.783         | 0.646          | 0.705       | 0.728          |
| hb-xlarge-ls960-ft | 0.670         | **0.719** | **0.722**    | 0.743        | **0.797** | **0.788**    | 0.741         | **0.788**  | **0.784**     | 0.693          | **0.734**   | **0.745**      |
| wlm-large          | 0.613         | 0.649     | 0.656        | 0.654        | 0.700     | 0.736        | 0.644         | 0.695      | 0.726         | 0.620          | 0.659       | 0.680          |
| wlm-base-plus      | 0.603         | 0.636     | 0.653        | 0.686        | 0.701     | 0.716        | 0.681         | 0.696      | 0.708         | 0.632          | 0.653       | 0.673          |
| wlm-base-plus-sv   | 0.649         | 0.641     | 0.656        | 0.697        | 0.713     | 0.716        | 0.687         | 0.698      | 0.714         | 0.667          | 0.664       | 0.680          |

</small></small>



## 📊 PCA-Based Intrinsic Analysis

We propose a novel intrinsic interpretability method based on **PCA of hidden representations**.

<img src="https://github.com/user-attachments/assets/4a62e417-7ff2-4023-971a-a7acc4df1867" width="400"/>


- **Conical (Wav2Vec2.0)**: emphasizes score continuity
- **V-shape (HuBERT)**: two-axis decision
- **S-shape (WavLM)**: diverse scoring factors

---

## 📎 Citation

If you use this repository or our findings, please cite:

```bibtex
@inproceedings{lee2024analysis,
  title={Analysis of Various Self-Supervised Learning Models for Automatic Pronunciation Assessment},
  author={Lee, Haeyoung and Kim, Sunhee and Chung, Minhwa},
  booktitle={2024 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages={1--6},
  year={2024},
  organization={IEEE}
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
