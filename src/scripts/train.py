import sys
sys.path.append("./")

import argparse
import os

parser = argparse.ArgumentParser(description="Train model from config")

parser.add_argument(
    "--config",
    type=str,
    help="Path to config file",
    required=True
)
parser.add_argument(
    "--device",
    default="0",
    type=str,
    help="Device index for CUDA_VISIBLE_DEVICES variable"
)

args = parser.parse_args()

# set device before src imports (see https://github.com/pytorch/pytorch/issues/9158)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from src.trainer.trainer import Trainer
from src.parser.parser import Parser

# parse args
trainer = Trainer(**Parser.parse_config(args))
trainer.train()
