import argparse
import pandas as pd
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings

parser = argparse.ArgumentParser(description='Embedding test')
parser.add_argument('input', type=Path, help='Input file with one query per line')
parser.add_argument('-o', '--output', type=Path, default='embeddings.csv', help='Output file')
parser.add_argument('--dry_run', action='store_true', help='Dry run (no embedding calls yet)')
args = parser.parse_args()

embeddings = OpenAIEmbeddings()

# Load from a txt file where each test string is a line into a list

with open(args.input, "r") as f:
    data = f.readlines()

# Remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]

# Build a pandas dataframe with text strings and their embedding vectors
dataset = []
for text in data:
    if args.dry_run:
        print(f"Text: {text}")
    else:
        dataset.append({
            "text": text,
            "embedding": embeddings.embed_query(text)
        })

df = pd.DataFrame(dataset)

# Save the dataframe to a csv file
df.to_csv(args.output, index=False)
