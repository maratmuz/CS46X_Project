import os 
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, TrainerCallback, AutoModelForSequenceClassification
from datetime import datetime
import torch
import gc
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import numpy as np
import argparse
import wandb

import matplotlib.pyplot as plt

from lib.dataloader import load_data
from lib.get_model import get_model


model_names = {
    "agro_nt": "InstaDeepAI/agro-nucleotide-transformer-1b"
}

task_names={
    'glycine_max': 'gene_exp.glycine_max', 
    'oryza_sativa': 'gene_exp.oryza_sativa', 
    'solanum_lycopersicum': 'gene_exp.solanum_lycopersicum', 
    'zea_mays': 'gene_exp.zea_mays', 
    'arabidopsis_thaliana': 'gene_exp.arabidopsis_thaliana'
}


class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.eval_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()


def compute_r2(eval_pred):
    predictions, labels = eval_pred

    # Squeeze the last dimension if it exists (for single output regression tasks)
    if predictions.ndim == 2 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    if labels.ndim == 2 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)

    r2 = r2_score(labels, predictions)
    return {"r2_score": r2}


def main():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="agro_nt", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="InstaDeepAI/plant-genomic-benchmark", help="Dataset name or path")
    parser.add_argument("--task_name", type=str, default="arabidopsis_thaliana", help="Task name within the dataset")
    parser.add_argument("--fine_tune_method", type=str, default="lora", help="Fine-tuning method to use (e.g., 'lora', 'full_fine_tuning'(not implemented yet))")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps. Overrides --num_epochs if set to a positive value.")

    parser.add_argument("--num_log_steps", type=int, default=100, help="Number of times to log training metrics during the entire training process (e.g., 100 means logging every 1% of the training steps)")
    parser.add_argument("--num_eval_steps", type=int, default=20, help="Number of times to evaluate the model on the validation set during the entire training process (e.g., 20 means evaluating every 5% of the training steps)")
    parser.add_argument("--num_save_steps", type=int, default=20, help="Number of times to save the model during the entire training process (e.g., 20 means saving every 5% of the training steps)")
    parser.add_argument("--output_dir", type=str, default="runs", help="Local directory to save the best model at the end of training")
    
    parser.add_argument("--report_to", type=str, default="none", help="The integration to report the results and logs to. Supported platforms are: 'tensorboard', 'wandb' and 'comet_ml'. Use 'none' for no logging.")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group name (e.g., for grouping runs by task)")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name (e.g., for identifying specific runs)")
    
    args = parser.parse_args()

    if args.report_to == "wandb":
        if args.wandb_project is None:
            args.wandb_project = f"{args.model_name}_seq2expr"
        if args.wandb_group is None:
            args.wandb_group = args.task_name
        if args.wandb_name is None:
            args.wandb_name = f"{args.model_name}_{args.task_name}_{now}" 
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "model_name": args.model_name,
                "dataset": args.dataset,
                "task_name": args.task_name,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "max_steps": args.max_steps,
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = model_names[args.model_name]
    task = task_names[args.task_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data = load_data(args.dataset, task_name=task, split="train", tokenizer=tokenizer)
    val_data = load_data(args.dataset, task_name=task, split="validation", tokenizer=tokenizer)
    test_data = load_data(args.dataset, task_name=task, split="test", tokenizer=tokenizer)
    
    num_labels = len(train_data[0]["labels"])
    
    model = get_model(model_name, num_labels, args.fine_tune_method)
    model.to(device) # Put the model on the GPU

    total_steps = args.max_steps if args.max_steps > 0 else (25000 * args.num_epochs) // args.batch_size

    output_dir = f"{args.output_dir}/{args.task_name}_{now}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=max(1, total_steps // args.num_log_steps),
        save_steps=max(1, total_steps // args.num_save_steps),
        eval_steps=max(1, total_steps // args.num_eval_steps),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        load_best_model_at_end=True,
        save_total_limit=1,
        learning_rate=5e-4,
        metric_for_best_model="r2_score",
        label_names=["labels"],
        report_to=args.report_to,
        skip_memory_metrics=False,

        # Memory efficiency settings
        bf16=True,        
        gradient_accumulation_steps= 1,
        gradient_checkpointing=True,      
        optim="adamw_torch_fused",  
    )

    # Starting Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        compute_metrics=compute_r2,
    )

    train_results = trainer.train()

    test_results = trainer.predict(test_data)
    test_r2 = test_results.metrics["test_r2_score"]
    print(f"R2 score on the test dataset: {test_r2}")
    if args.report_to == "wandb":
        wandb.log({"test/r2_score": test_r2})
        wandb.finish()

    # Save training parameters and results to a text file for easy reference
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write("=== Training Parameters ===\n")
        f.write(f"model_name:       {args.model_name}\n")
        f.write(f"dataset:          {args.dataset}\n")
        f.write(f"task_name:        {args.task_name}\n")
        f.write(f"fine_tune_method: {args.fine_tune_method}\n")
        f.write(f"batch_size:       {args.batch_size}\n")
        f.write(f"num_epochs:       {args.num_epochs}\n")
        f.write(f"learning_rate:    5e-4\n")
        f.write(f"timestamp:        {now}\n")
        f.write("\n=== Results ===\n")
        f.write(f"test_r2_score:    {test_r2:.6f}\n")


if __name__ == "__main__":
    main()