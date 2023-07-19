from tqdm import tqdm
import logging

import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.datasets import create_datasets
from src.parser import Parser

class Trainer:
    """Class for training the model in the domain generalization mode.
    """

    def __init__(
            self,
            config,
            device,
            model_name,
            optimizer_config,
            scheduler_config,
            dataset,
            num_epochs,
            batch_size,
            run_id):
        self.config = config

        self.device = device

        self.model_name = model_name

        self.dataset = dataset
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.run_id = run_id
        self.checkpoint_dir = f"saved/{self.run_id}/"

    def calculate_metrics(self, true_labels, pred_labels):
        res = {}
        res["accuracy"] = accuracy_score(true_labels, pred_labels)
        for metric_name, metric in zip(["precision", "recall", "f1"], [precision_score, recall_score, f1_score]):
            res[metric_name] = metric(pred_labels, true_labels, average="weighted")
        return res 

    def init_training(self):
        self.tokenizer = Parser.init_tokenizer(self.model_name)
        self.dataset["kwargs"]["tokenizer"] = self.tokenizer
        self.model = Parser.init_model(self.model_name, self.dataset["kwargs"]["num_classes"], self.device)
        self.optimizer = Parser.init_optimizer(
            self.optimizer_config, self.model)
        self.scheduler = Parser.init_scheduler(
            self.scheduler_config, self.optimizer)

    def train_epoch_model(self, loader):
        self.model.train()
        loss_sum = 0
        true_labels, pred_labels = [], []
        pbar = tqdm(loader)
        for input_ids, attention_mask, labels in pbar:
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
            logits = self.model(input_ids, attention_mask)["logits"]
            loss = self.loss_function(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * input_ids.shape[0]
            true_labels += labels.tolist()
            pred_labels += F.softmax(logits, dim=-1).argmax(dim=-1).tolist()

        res = self.calculate_metrics(true_labels, pred_labels)
        res["loss"] = loss_sum / len(loader.dataset)
        return res

    def inference_epoch_model(self, loader):
        with torch.inference_mode():
            self.model.eval()
            loss_sum = 0
            true_labels, pred_labels = [], []
            pbar = tqdm(loader)
            for input_ids, attention_mask, labels in pbar:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                logits = self.model(input_ids, attention_mask)["logits"]
                loss = self.loss_function(logits, labels)
                loss_sum += loss.item() * input_ids.shape[0]
                true_labels += labels.tolist()
                pred_labels += F.softmax(logits, dim=-1).argmax(dim=-1).tolist()

        res = self.calculate_metrics(true_labels, pred_labels)
        res["loss"] = loss_sum / len(loader.dataset)
        return res


    def create_loaders(self):
        train_dataset, val_dataset, test_dataset = create_datasets(self.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4)

        val_loader, test_loader = [torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4) for dataset in [val_dataset, test_dataset]]
        return train_loader, val_loader, test_loader

    def train(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        self.init_training()
        train_loader, val_loader, test_loader = self.create_loaders()
        max_val_accuracy = 0

        for i in range(1, self.num_epochs + 1):
            logging.info(f"Start epoch number {i}")
            train_metrics = self.train_epoch_model(train_loader)
            logging.info(f"Epoch number {i} is over. Metrics on train: {train_metrics}")
            val_metrics = self.inference_epoch_model(val_loader)
            logging.info(f"Epoch number {i} is over. Metrics on val: {val_metrics}")
            
            if val_metrics["f1"] > max_val_accuracy:
                max_val_accuracy = val_metrics["f1"]
                self.save_checkpoint()

            if self.scheduler is not None:
                self.scheduler.step()

        self.load_checkpoint()
        test_metrics = self.inference_epoch_model(test_loader)
        logging.info(f"ERM training. Results on test: {test_metrics}")

    def save_checkpoint(self):
        state = {
            "name": self.model_name,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else [],
            "config": self.config}
        path = f"{self.checkpoint_dir}checkpoint_name_{state['name']}_best.pth"
        torch.save(state, path)

    def load_checkpoint(self):
        self.model = Parser.init_model(self.model_name, self.dataset["kwargs"]["num_classes"], self.device)
        model_path = f"{self.checkpoint_dir}checkpoint_name_{self.model_name}_best.pth"
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])
