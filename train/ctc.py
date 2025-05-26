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

    
class PronunciationScoringModel(nn.Module):
    def __init__(self, model_name, model_save_path):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)        
        self.score_predictor = nn.Linear(32, 4)
        
    def forward(self, input_values, labels=None):
        outputs = self.model(input_values=input_values)
        logits = outputs.logits
        pooled_output = logits.mean(dim=1)
        scores = self.score_predictor(pooled_output).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(scores, labels.float())
        return {"loss": loss, "scores": scores}

def prepare_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune SSL models for pronunciation scoring task.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--model_name", type=str, default="facebook/hubert-xlarge-ll60k", help="Model identifier from Huggingface Model Hub.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs.")
    parser.add_argument("--save_dir", type=str, default="/data2/haeyoung/0303/ctc/model_saved", help="Directory to save the trained model.")
    parser.add_argument("--freeze_feature_extractor", action="store_true", help="Whether to freeze the feature extractor layers of the model.")
    args = parser.parse_args()
    args.exp_name = f"{args.model_name}_lr{args.learning_rate}_epochs{args.num_train_epochs}_batch{args.batch_size}"
    args.save_dir_path = os.path.join(args.save_dir, args.exp_name)
    args.save_log_path = os.path.join(args.save_dir, "logs", args.exp_name)
    
    os.makedirs(args.save_dir_path, exist_ok=True)
    os.makedirs(args.save_log_path, exist_ok=True)
    
    with open(os.path.join(args.save_dir_path, "args.json"), "w") as args_file:
        json.dump(vars(args), args_file, indent=4)
    
    return args


def prepare_dataset(batch, feature_extractor):
    array = batch["audio"]["array"]
    input_values_tensor = feature_extractor(array, sampling_rate=16000).input_values[0]
    batch["input_values"] = torch.tensor(input_values_tensor, dtype=torch.float)
    # set pronunciation scores as labels
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


def prepare_trainer(args, model, feature_extractor, train_ds, test_ds):
    
    if args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=args.save_dir_path,
        logging_dir=args.save_log_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        log_level="info",
        save_strategy="epoch",
        save_total_limit=3,
        report_to=["tensorboard"]
    )
    
    with open(os.path.join(args.save_dir_path, "trainer_args.json"), "w") as args_file:
        json.dump(training_args.to_dict(), args_file, ensure_ascii=False)
        

    data_collator = DataCollatorForAPA(padding_value=0.0)        
        
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )
    return trainer

def main():
        
    args = prepare_arguments()
    print(args)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=args.save_log_path + "/logging.log",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    # MODIFY PATHS BELOW TO YOUR Speechocean762 DATASET
    train_ocean_ds_path = "/path/to/your/speechocean_train_ds"   # <-- MODIFY THIS PATH
    test_ocean_ds_path = "/path/to/your/speechocean_test_ds"     # <-- MODIFY THIS PATH

    ds_train = load_from_disk(train_ocean_ds_path).map(lambda batch: prepare_dataset(batch, feature_extractor))
    ds_test = load_from_disk(test_ocean_ds_path).map(lambda batch: prepare_dataset(batch, feature_extractor))
    
    model = PronunciationScoringModel(args.model_name, args.save_dir_path)
    trainer = prepare_trainer(args, model, feature_extractor, ds_train, ds_test)
    
    # train
    train_res = trainer.train()
    trainer.save_model()
    trainer.save_state()
    metrics = train_res.metrics
    metrics["train_samples"] = len(ds_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(ds_test)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    with open(args.save_log_path + "/training_log.log", "w") as f:
        for obj in trainer.state.log_history:
            f.write(str(obj) + "\n")

    print("- Training complete.")
    
    torch.save(model.state_dict(), os.path.join(args.save_dir_path, "model_weights.pt"))
    bin_path = os.path.join(args.save_dir_path, "finetuned_pytorch_model.bin")
    torch.save(model.state_dict(), bin_path)

if __name__ == "__main__":
    main()

   