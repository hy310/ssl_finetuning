import argparse
import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Union
from datasets import load_dataset, load_from_disk, Audio
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
from transformers import (
    EarlyStoppingCallback, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, 
    TrainingArguments, Trainer, AutoConfig, AutoModelForCTC, Wav2Vec2PhonemeCTCTokenizer, AutoFeatureExtractor, Wav2Vec2Model, HubertModel, WavLMModel
)

def prepare_dataset(batch, feature_extractor):
    array = batch["audio"]["array"]
    input_values_tensor = feature_extractor(array, sampling_rate=16000).input_values[0]
    batch["input_values"] = torch.tensor(input_values_tensor, dtype=torch.float)
    # Use multiple pronunciation scores as targets
    batch["labels"] = torch.tensor([batch["accuracy"], batch["fluency"], batch["prosodic"], batch["total"]], dtype=torch.float)
    return batch

class DataCollatorForAPA:
    def __init__(self, padding_value=0.0):
        self.padding_value = padding_value

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        input_values = [torch.tensor(feature['input_values'], dtype=torch.float) for feature in features]
        input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=self.padding_value)
        labels = torch.tensor([feature['labels'] for feature in features], dtype=torch.float)

        batch = {
            "input_values": input_values_padded,
            "labels": labels
        }
        
        return batch


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    mse = mean_squared_error(labels, preds, multioutput='raw_values')
    pcc = [np.corrcoef(labels[:, i], preds[:, i])[0, 1] if not np.isnan(np.corrcoef(labels[:, i], preds[:, i])[0, 1]) else 0 for i in range(preds.shape[1])]

    metrics = {}
    for i, (mse_val, pcc_val) in enumerate(zip(mse, pcc)):
        metrics[f"mse_{i}"] = mse_val
        metrics[f"pcc_{i}"] = pcc_val

    return metrics

def main():
        
    args = prepare_arguments()
    print(args)

    # === MODIFY BELOW PATHS AS NEEDED ===
    test_ocean_ds_path = "/path/to/your/preprocess/speechocean_test_ds"  # <-- Update this path
    PTM = "/path/to/your/model_weights.pt"   # <-- Update this path
    # ====================================
    print("- MODEL:", PTM)
    ds_test = load_from_disk(test_ocean_ds_path).map(lambda batch: prepare_dataset(batch, feature_extractor))
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(PTM)
    if config.model_type == "hubert":
        print("-Using Hubert")
        model = HubertModel.from_pretrained(PTM)
    elif config.model_type == "wavlm":
        print("-Using WavLM")
        model = WavLMModel.from_pretrained(PTM)
    else:
        print("-Using Wav2Vec2")
        model = Wav2Vec2Model.from_pretrained(PTM)
    
    data_collator = DataCollatorForAPA(padding_value=0.0)        
    training_args = TrainingArguments(
        output_dir=".",
        group_by_length=True,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=4,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics)
    metrics = trainer.evaluate()