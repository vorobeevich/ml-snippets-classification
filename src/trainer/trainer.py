from tqdm import tqdm
import logging
from copy import deepcopy

import torch.utils.data
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.datasets import create_datasets
from src.parser import Parser
import src.datasets
from src.utils import init_object

from transformers.optimization import get_linear_schedule_with_warmup

class Trainer:
    """Class for training the model in the domain generalization mode.
    """

    def __init__(
            self,
            config,
            device,
            model_name,
            optimizer_config,
            dataset,
            train_marks,
            additional_data,
            num_epochs,
            batch_size,
            run_id):
        self.config = config

        self.device = device

        self.model_name = model_name

        self.dataset = dataset
        self.train_marks = train_marks
        self.additional_data = additional_data
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.run_id = run_id
        self.checkpoint_dir = f"saved/{self.run_id}/"

    def calculate_metrics(self, true_labels, pred_labels):
        res = {}
        res["accuracy"] = accuracy_score(true_labels, pred_labels)
        for metric_name, metric in zip(["recall", "f1"], [recall_score, f1_score]):
            res[metric_name] = metric(true_labels, pred_labels, average="weighted")
        res["precision"] = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
        return res 

    def init_training(self):
        self.tokenizer = Parser.init_tokenizer(self.model_name)
        self.dataset["kwargs"]["tokenizer"] = self.tokenizer
        self.model = Parser.init_model(self.model_name, self.dataset["kwargs"]["num_classes"], self.device)
        self.optimizer = Parser.init_optimizer(
            self.optimizer_config, self.model)

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
            self.scheduler.step()
            loss_sum += loss.item() * input_ids.shape[0]
            true_labels += labels.tolist()
            pred_labels += logits.argmax(dim=-1).tolist()

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
                pred_labels += logits.argmax(dim=-1).tolist()

        res = self.calculate_metrics(true_labels, pred_labels)
        res["loss"] = loss_sum / len(loader.dataset)
        return res


    def create_loaders(self):
        train_dataset, val_dataset, test_dataset = create_datasets(self.dataset)
        assert self.train_marks != [] or self.additional_data, "You must choose data to train"
        if self.train_marks != [5] and self.train_marks != []:  
            new_train_dataset = deepcopy(self.dataset)
            new_train_dataset["kwargs"]["marks"] = [elem for elem in self.train_marks if elem != 5]
            new_train_dataset["kwargs"]["dataset_type"] = "all"
            new_train_dataset = init_object(src.datasets, new_train_dataset)
            if 5 in self.train_marks:
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, new_train_dataset])
            else:
                train_dataset = new_train_dataset
        
        if self.additional_data:
            additional_dataset = deepcopy(self.dataset)
            additional_dataset["kwargs"]["path"] = self.additional_data
            additional_dataset["kwargs"]["marks"] = [1, 2, 3, 4, 5]
            additional_dataset["kwargs"]["dataset_type"] = "all"
            additional_dataset = init_object(src.datasets, additional_dataset)
            if self.train_marks == []:
                train_dataset = additional_dataset
            else:
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, additional_dataset])

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

        with open(f"{self.checkpoint_dir}logs.txt", "w") as f:
            print("Start training. \nF1 history:", file=f)

        self.init_training()
        train_loader, val_loader, test_loader = self.create_loaders()
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(len(train_loader.dataset) / self.batch_size) * int(self.num_epochs / 10), num_training_steps = int(len(train_loader.dataset) / self.batch_size) * self.num_epochs)
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

            with open(f"{self.checkpoint_dir}logs.txt", "a") as f:
                print(f"Epoch number {i}. Train f1: {train_metrics['f1']}. Val f1: {val_metrics['f1']}", file=f)

        self.load_checkpoint()
        test_metrics = self.inference_epoch_model(test_loader)
        logging.info(f"ERM training. Results on test: {test_metrics}")
        with open(f"{self.checkpoint_dir}logs.txt", "a") as f:
            print(f"Finish training.\n Test f1: {test_metrics['f1']}. Test accuracy: {test_metrics['accuracy']}.", file=f)
            print(f"Test precision: {test_metrics['precision']}. Test recall: {test_metrics['recall']}", file=f)


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
