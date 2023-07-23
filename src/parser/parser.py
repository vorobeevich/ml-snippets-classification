import os
import yaml
import argparse
from datetime import datetime
from copy import deepcopy
import typing as tp

import torch

from src.utils import fix_seed, get_device, init_object
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import BertForSequenceClassification, BertTokenizer


class Parser:
    """Static class (a set of methods for which inheritance is possible) for parsing command line arguments
    and model training configuration.
    """

    @staticmethod
    def parse_config(args: argparse.Namespace) -> dict[str, tp.Any]:
        """Parse command line parameters and config_yaml parameters (dataset, model architecture,
        training parameters, augmentations, etc).

        Args:
            args (argparse.Namespace): args from cl parser (path to config_yaml, path to checkpoint, GPU device).
        """

        # read config_yaml from path
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)

        # fix random seed for reproducibility
        fix_seed(config["seed"])

        # init params for trainer
        trainer_params = dict()

        # send some params directly to trainer
        for param in ["model_name", "dataset", "train_marks", "additional_data", "num_epochs", "batch_size", "run_id"]:
            trainer_params[param] = config[param]
        trainer_params["dataset"]["kwargs"]["seed"] = config["seed"]

        trainer_params["optimizer_config"] = config["optimizer"]

        # init run_id
        if config["run_id"] is None:
            # use timestamp as default run-id
            config["run_id"] = datetime.now().strftime(r"%m%d_%H%M%S")

        # make checkpoint_dir
        if not os.path.exists("saved/"):
            os.makedirs("saved/")
        if not os.path.exists(f'saved/{config["run_id"]}/'):
            os.makedirs(f'saved/{config["run_id"]}/')

        # save device id in config
        config["device_id"] = args.device

        # save yaml version of config
        trainer_params["config"] = config

        # init device for training on it
        trainer_params["device"] = get_device()
        return trainer_params

    @staticmethod
    def init_tokenizer(model_name: str):
        if model_name == "codebert":
            tokenizer = RobertaTokenizer.from_pretrained(
                "microsoft/codebert-base")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer

    @staticmethod
    def init_model(model_name: str, num_classes: int, device: torch.device):
        # init model params
        if model_name == "codebert":
            model = RobertaForSequenceClassification.from_pretrained(
                "microsoft/codebert-base", num_labels=num_classes)
        else:
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_classes)

        # prepare for GPU training
        model.to(device)

        return model

    @staticmethod
    def init_optimizer(config, model):
        # make deepcopy to not corrupt dict
        optimizer = deepcopy(config)

        # init optimizer with model params
        optimizer["kwargs"].update(params=model.parameters())
        optimizer = init_object(torch.optim, optimizer)

        return optimizer
