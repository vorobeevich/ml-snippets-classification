import sys
sys.path.append("./")

import argparse
import os
import logging 
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Split data from config")

parser.add_argument(
    "--data_path",
    type=str,
    help="Path to data",
    required=True
)

parser.add_argument(
    "--num_classes",
    type=int,
    help="Num classes in dataset",
    required=True
)

parser.add_argument(
    "--marks",
    nargs="*",  
    type=int,
    help="Marks values for dataset to use",
    required=True
)

parser.add_argument(
    "--max_lines",
    type=int,
    help="Max lines in snippet tp split",
    required=True
)

parser.add_argument(
    "--model_name",
    type=str,
    help="Name of model",
    required=True
)

parser.add_argument(
    "--model_path",
    type=str,
    help="Path to model",
    required=True
)

parser.add_argument(
    "--result_path",
    type=str,
    help="Path to result data",
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
from src.splitter import Splitter
from src.preprocessing import preprocess_snippet
from src.datasets import load_data
from src.utils import get_device
from src.parser import Parser
from transformers import RobertaTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

device = get_device()
logging.info(f"DEVICE SELECTED : {device}")

logging.info("START LOADING MODEL...")
model = Parser.init_model(args.model_name, args.num_classes, device)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model.to(device)

logging.info("START LOADING DATA...")
input_data = load_data(args.data_path, args.num_classes, args.marks)

logging.info("START SPLITTING DATA...")
output_data = pd.DataFrame(columns=input_data.columns)
splitter = Splitter(model, tokenizer, device)
splitter.set_max_lines(args.max_lines)
pbar = tqdm(list(input_data.iterrows()))
split_counter = 0
for ind, row in pbar:
    snippet = preprocess_snippet(row['code_block'], 'str')
    split = splitter.split_snippet(snippet)
    sub_snippets = splitter.split_by_indexes(snippet, split)
    for sub_snippet in sub_snippets:
        new_row = row
        new_row['code_block'] = sub_snippet
        new_row['graph_vertex_id'] = splitter.predict_subsnippet_class(
            sub_snippet)
        output_data.loc[output_data.shape[0]] = new_row
    if len(sub_snippets) > 1:
        split_counter += 1
    pbar.set_description("ALREADY SPLITTED: %s" % split_counter)

logging.info("START SAVING DATA...")
output_data.to_csv(args.result_path, index=False)
