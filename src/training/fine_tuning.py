from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
import mlflow

from ..config.settings import settings


class ContentDataset(Dataset):
    """Dataset for fine-tuning on extracted content."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        
        return item


class QADataset(Dataset):
    """Dataset for question-answering fine-tuning."""
    
    def __init__(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str],
        tokenizer=None,
        max_length: int = 512
    ):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        # Format: Question: {q} Context: {c}
        input_text = f"Question: {self.questions[idx]}\nContext: {self.contexts[idx]}"
        target_text = self.answers[idx]
        
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


class FineTuningPipeline:
    """Enterprise-grade fine-tuning with LoRA, MLflow tracking, and evaluation."""
    
    def __init__(
        self,
        base_model: str = "google/flan-t5-base",
        output_dir: Path = None,
        use_lora: bool = True,
        use_mlflow: bool = True
    ):
        self.base_model = base_model
        self.output_dir = output_dir or settings.MODELS_DIR / "fine_tuned"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_lora = use_lora
        self.use_mlflow = use_mlflow
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Fine-Tuning Pipeline on {self.device}")
        logger.info(f"Base model: {base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup MLflow
        if use_mlflow:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("learning_content_finetuning")
    
    def prepare_classification_model(self, num_labels: int):
        """Prepare model for topic/concept classification."""
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=num_labels,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model.to(self.device)
    
    def prepare_qa_model(self):
        """Prepare model for question answering."""
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model.to(self.device)
    
    def train_topic_classifier(
        self,
        texts: List[str],
        labels: List[int],
        label_names: List[str],
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """Train a topic classifier on extracted content."""
        
        logger.info(f"Training topic classifier with {len(texts)} samples")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Create datasets
        train_dataset = ContentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = ContentDataset(val_texts, val_labels, self.tokenizer)
        
        # Prepare model
        model = self.prepare_classification_model(len(label_names))
        
        # Training arguments
        run_name = f"topic_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / run_name),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=10,
            fp16=self.device == "cuda",
            report_to="mlflow" if self.use_mlflow else "none"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        if self.use_mlflow:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "base_model": self.base_model,
                    "num_labels": len(label_names),
                    "label_names": label_names,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "use_lora": self.use_lora
                })
                
                train_result = trainer.train()
                
                mlflow.log_metrics({
                    "train_loss": train_result.training_loss,
                    "train_samples": len(train_texts),
                    "val_samples": len(val_texts)
                })
        else:
            train_result = trainer.train()
        
        # Save model
        model_path = self.output_dir / run_name / "final_model"
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        # Save label mapping
        label_mapping = {i: name for i, name in enumerate(label_names)}
        with open(model_path / "label_mapping.json", "w") as f:
            json.dump(label_mapping, f)
        
        logger.info(f"Model saved to {model_path}")
        
        return {
            "model_path": str(model_path),
            "train_loss": train_result.training_loss,
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "labels": label_names
        }
    
    def train_qa_model(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str],
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 3e-5
    ) -> Dict[str, Any]:
        """Train a QA model on generated question-answer pairs."""
        
        logger.info(f"Training QA model with {len(questions)} samples")
        
        # Split data
        indices = list(range(len(questions)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_ctx = [contexts[i] for i in train_idx]
        train_q = [questions[i] for i in train_idx]
        train_a = [answers[i] for i in train_idx]
        
        val_ctx = [contexts[i] for i in val_idx]
        val_q = [questions[i] for i in val_idx]
        val_a = [answers[i] for i in val_idx]
        
        # Create datasets
        train_dataset = QADataset(train_ctx, train_q, train_a, self.tokenizer)
        val_dataset = QADataset(val_ctx, val_q, val_a, self.tokenizer)
        
        # Prepare model
        model = self.prepare_qa_model()
        
        # Training arguments
        run_name = f"qa_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / run_name),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
            fp16=self.device == "cuda",
            report_to="mlflow" if self.use_mlflow else "none"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        train_result = trainer.train()
        
        # Save
        model_path = self.output_dir / run_name / "final_model"
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        logger.info(f"QA Model saved to {model_path}")
        
        return {
            "model_path": str(model_path),
            "train_loss": train_result.training_loss,
            "train_samples": len(train_q),
            "val_samples": len(val_q)
        }
    
    def generate_training_data(
        self,
        processed_contents: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate training data from processed content."""
        
        contexts = []
        questions = []
        answers = []
        
        for content in processed_contents:
            text = content.get("raw_text", "")
            flashcards = content.get("flashcards", [])
            
            for fc in flashcards:
                # Use flashcards as QA pairs
                contexts.append(text[:2000])
                questions.append(fc.get("question", ""))
                answers.append(fc.get("answer", ""))
        
        return contexts, questions, answers
