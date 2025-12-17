
from glob import glob
import pandas as pd
import argparse
from os.path import join as opj

parser = argparse.ArgumentParser()
parser.add_argument("--txt_dir", type=str, default="./results/sample_caption")
parser.add_argument("--save_csv_p", type=str, default="./results/sample_caption.csv")
args = parser.parse_args()

txt_ps = sorted(glob(opj(args.txt_dir, "*.txt")))
split_idx = -1

data = dict()
data["file_name"] = []
data["text"] = []

for txt_p in txt_ps:
    with open(txt_p, "r") as f: 
        text = f.readlines()[0].strip()

    file_name = "/".join(txt_p.split("/")[split_idx:]).replace(".txt", ".mp4")   
    data["file_name"].append(file_name)
    data["text"].append(text)
    
df = pd.DataFrame(data)
df.to_csv(args.save_csv_p, index=False)