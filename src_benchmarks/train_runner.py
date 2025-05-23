import logging
import os
import json
import pickle
from pathlib import Path

import optuna
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from src_benchmarks.base_runner import BaseRunner
from src.metrics import (
    OfflineMetrics,
    Recall,
    Precision,
    MAP,
    NDCG,
    HitRate,
    MRR,
    Coverage,
    Surprisal,
)
from src.data.nn import SequenceTokenizer
from src.metrics.torch_metrics_builder import metrics_to_df
from src.models.nn.sequential import SasRec
from src.models.nn.optimizer_utils import FatOptimizerFactory
from src.models.nn.sequential.callbacks import (
    ValidationMetricsCallback,
    PandasPredictionCallback,
)
from src.models.nn.sequential.postprocessors import RemoveSeenItems
from src.models.nn.sequential.sasrec import (
    SasRecTrainingDataset,
    SasRecValidationDataset,
    SasRecPredictionDataset,
)


class TrainRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.item_count = None
        self.raw_test_gt = None
        self.seq_val_dataset = None
        self.seq_test_dataset = None
        self.popularity_distribution = None

        # Loggers
        self.log_dir = (Path(config["paths"]["log_dir"]) / self.dataset_name / self.model_save_name)
        self.csv_logger = CSVLogger(save_dir=self.log_dir / "csv_logs")
        self.tb_logger = TensorBoardLogger(save_dir=self.log_dir / "tb_logs")

        self._check_paths()

    def _check_paths(self):
        """Ensure all required directories exist."""
        required_paths = [
            self.config["paths"]["log_dir"],
            self.config["paths"]["checkpoint_dir"],
            self.config["paths"]["results_dir"],
        ]
        for path in required_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _initialize_model(self, trial=None):
        """Initialize the model based on configuration or Optuna trial parameters."""
        model_config = {
            "tensor_schema": self.tensor_schema,
        }

        
        optimizer_factory = FatOptimizerFactory(
            learning_rate=self.model_cfg["training_params"]["learning_rate"],
            weight_decay=self.model_cfg["training_params"].get("weight_decay", 0.0),
        )

        model_config.update(self.model_cfg["model_params"])

        if "sasrec" in self.model_name.lower():
            print(self.popularity_distribution)
            return SasRec(
                **model_config,
                optimizer_factory=optimizer_factory,
                popularity_distribution=self.popularity_distribution,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def _prepare_dataloaders(
        self,
        seq_train_dataset,
        seq_validation_dataset,
        seq_validation_gt,
        seq_test_dataset,
    ):
        """Initialize dataloaders for training, validation, and testing."""
        logging.info("Preparing dataloaders...")

        dataset_mapping = {
            "sasrec": (
                SasRecTrainingDataset,
                SasRecValidationDataset,
                SasRecPredictionDataset,
            ),
        }

        pad_idx = 0 # self.tensor_schema.item_id_features.item().cardinality

        datasets = dataset_mapping.get(self.model_name.lower())
        if not datasets:
            raise ValueError(
                f"Unsupported model type for dataloaders: {self.model_name}"
            )

        TrainingDataset, ValidationDataset, PredictionDataset = datasets
        common_params = {
            "batch_size": self.model_cfg["training_params"]["batch_size"],
            "num_workers": self.model_cfg["training_params"]["num_workers"],
            "pin_memory": True,
        }

        train_dataloader = DataLoader(
            dataset=TrainingDataset(
                seq_train_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
                padding_value=pad_idx,
            ),
            shuffle=True,
            **common_params,
        )
        val_dataloader = DataLoader(
            dataset=ValidationDataset(
                seq_validation_dataset,
                seq_validation_gt,
                seq_train_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
                padding_value=pad_idx,
            ),
            **common_params,
        )
        val_pred_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_validation_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
                padding_value=pad_idx,
            ),
            **common_params,
        )
        prediction_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_test_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            **common_params,
        )

        return (
            train_dataloader,
            val_dataloader,
            val_pred_dataloader,
            prediction_dataloader,
        )

    def _load_dataloaders(self):
        """Loads data and prepares dataloaders."""
        logging.info("Preparing datasets for training.")
        train_events, validation_events, validation_gt, test_events, test_gt = (
            self.load_data()
        )
        self.validation_gt = validation_gt
        self.test_events = test_events
        self.raw_test_gt = test_gt

        (
            train_dataset,
            train_val_dataset,
            val_dataset,
            val_gt_dataset,
            test_dataset,
            test_gt_dataset,
        ) = self.prepare_datasets(
            train_events, validation_events, validation_gt, test_events, test_gt
        )
        self.item_count = train_dataset.item_count

        (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        ) = self.prepare_seq_datasets(
            train_dataset,
            train_val_dataset,
            val_dataset,
            val_gt_dataset,
            test_dataset,
            test_gt_dataset,
        )
        self.seq_val_dataset = seq_validation_dataset
        self.seq_test_dataset = seq_test_dataset

        return self._prepare_dataloaders(
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def calculate_metrics(self, predictions, ground_truth, test_events=None):
        """Calculate and return the desired metrics based on the predictions."""
        top_k = self.config["metrics"]["ks"]
        base_metrics = [
            Recall(top_k),
            Precision(top_k),
            MAP(top_k),
            NDCG(top_k),
            MRR(top_k),
            HitRate(top_k),
        ]

        diversity_metrics = []
        if test_events is not None:
            diversity_metrics = [
                Coverage(top_k),
                Surprisal(top_k),
            ]

        all_metrics = base_metrics + diversity_metrics
        metrics_results = OfflineMetrics(
            all_metrics, 
            query_column=self.user_column, 
            item_column=self.item_column, 
            rating_column="score",
        )(
            predictions,
            ground_truth,
            test_events,
        )

        return metrics_to_df(metrics_results)

    def save_model(self, trainer, best_model):
        """Save the best model checkpoint to the specified directory."""
        save_path = os.path.join(
            self.config["paths"]["checkpoint_dir"],
            f"{self.model_save_name}_{self.dataset_name}",
        )
        torch.save(
            {
                "model_state_dict": best_model.state_dict(),
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),
                "config": self.model_cfg,
            },
            f"{save_path}/{self.model_save_name}_checkpoint.pth",
        )

        self.tokenizer.save(f"{save_path}/sequence_tokenizer")
        logging.info(f"Best model saved at: {save_path}")

    def _save_allocated_memory(self):
        devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device=devices[0]) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device=devices[0]) / 1024**3  # GB
        torch.cuda.reset_peak_memory_stats()

        data = {
            'allocated_memory': [allocated],
            'max_allocated_memory': [max_allocated]
        }
        df = pd.DataFrame(data)

        df.to_csv(os.path.join(
            self.csv_logger.log_dir,
            "memory_stats.csv"
        ), index=False)

        logging.info(f"Allocated memory: {allocated} GB")
        logging.info(f"Max allocated memory: {max_allocated} GB")    

    def run(self):
        """Execute the training pipeline."""
        train_dataloader, val_dataloader, val_pred_dataloader, prediction_dataloader = (
            self._load_dataloaders()
        )
        
        logging.info("Initializing model...")
        model = self._initialize_model()

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.config["paths"]["checkpoint_dir"],
                f"{self.model_save_name}_{self.dataset_name}",
            ),
            save_top_k=1,
            verbose=True,
            monitor="ndcg@10",
            mode="max",
        )

        early_stopping = EarlyStopping(
            monitor="ndcg@10",
            patience=self.model_cfg["training_params"]["patience"],
            mode="max",
            verbose=True,
        )

        validation_metrics_callback = ValidationMetricsCallback(
            metrics=self.config["metrics"]["types"],
            ks=self.config["metrics"]["ks"],
            item_count=self.item_count,
            postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
        )

        profiler = SimpleProfiler(
            dirpath=self.csv_logger.log_dir, filename="simple_profiler"
        )

        devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
        trainer = L.Trainer(
            max_epochs=self.model_cfg["training_params"]["max_epochs"],
            callbacks=[
                checkpoint_callback,
                early_stopping,
                validation_metrics_callback,
            ],
            logger=[self.csv_logger, self.tb_logger],
            profiler=profiler,
            precision=self.model_cfg["training_params"]["precision"],
            devices=devices,
        )

        logging.info("Starting training...")
        if self.config["mode"]["profiler"]["enabled"]:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                trainer.fit(model, train_dataloader, val_dataloader)
            logging.info(
                prof.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=self.config["mode"]["profiler"].get("row_limit", 10),
                )
            )
            prof.export_chrome_trace(
                os.path.join(
                    self.config["paths"]["log_dir"],
                    f"{self.model_save_name}_{self.dataset_name}_profile.json",
                )
            )
        else:
            trainer.fit(model, train_dataloader, val_dataloader)
            self._save_allocated_memory()

        if self.model_name.lower() == "sasrec":
            best_model = SasRec.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )
        self.save_model(trainer, best_model)

        logging.info("Evaluating on val set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column=self.user_column,
            item_column=self.item_column,
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
        )
        L.Trainer(
            callbacks=[pandas_prediction_callback],
            inference_mode=True,
            devices=devices,
            precision=self.model_cfg["training_params"]["precision"]
        ).predict(
            best_model, dataloaders=val_pred_dataloader, return_predictions=False
        )

        result = pandas_prediction_callback.get_result()
        recommendations = (
            self.tokenizer.query_and_item_id_encoder.inverse_transform(result)
        )
        val_metrics = self.calculate_metrics(recommendations, self.validation_gt)
        logging.info(val_metrics)
        recommendations.to_parquet(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.model_save_name}_{self.dataset_name}_val_preds.parquet",
            ),
        )
        val_metrics.to_csv(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.model_save_name}_{self.dataset_name}_val_metrics.csv",
            ),
        )

        logging.info("Evaluating on test set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column=self.user_column,
            item_column=self.item_column,
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_test_dataset)],
        )
        L.Trainer(
            callbacks=[pandas_prediction_callback],
            inference_mode=True,
            devices=devices,
            precision=self.model_cfg["training_params"]["precision"]
        ).predict(best_model, dataloaders=prediction_dataloader, return_predictions=False)

        result = pandas_prediction_callback.get_result()
        recommendations = (self.tokenizer.query_and_item_id_encoder.inverse_transform(result))
        test_metrics = self.calculate_metrics(recommendations, self.raw_test_gt, self.test_events)
        logging.info(test_metrics)
        recommendations.to_parquet(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.model_save_name}_{self.dataset_name}_test_preds.parquet",
            ),
        )
        test_metrics.to_csv(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.model_save_name}_{self.dataset_name}_test_metrics.csv",
            ),
        )
