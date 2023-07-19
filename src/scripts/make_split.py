import torch
from tqdm import tqdm
import pandas as pd
from splitter import Splitter
from utils.preprocessing import preprocess_snippet
from utils.load_data import load_data
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE SELECTED : ", device)

print("START LOADING MODEL...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(
    "models/code_bert", num_labels=90)
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("MODEL SUCCESSFULLY LOADED")


print("START LOADING DATA...")
input_data = load_data("data/fixed_data_20220412.csv", [3, 4])
print("DATA SUCCESSFULLY LOADED")
print("START SPLITTING DATA...")

output_data = pd.DataFrame(columns=input_data.columns)
splitter = Splitter(model, tokenizer, device)

pbar = tqdm(input_data.iterrows())
split_counter = 0
for ind, row in pbar:
    snippet = preprocess_snippet(row['code_block'])
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

print("\nSTART SAVING DATA...")
output_data.to_csv(
    "auto_split/data/fixed_data_20220412_after_split.csv",
    index=False)
print("DATA SUCCESSFULLY SAVED")
