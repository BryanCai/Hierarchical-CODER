import pandas as pd
from pathlib import Path



if __name__ == "__main__":
    tree_dir = Path("D:/Projects/CODER/Hierarchical-CODER/data/cleaned/all")
    save_dir = Path("D:/Projects/CODER/Hierarchical-CODER/data/cleaned")
    tree_subdirs = [f for f in tree_dir.iterdir() if f.is_dir()]
    for tree_subdir in tree_subdirs:
        tree_path = tree_subdir/"hierarchy.csv"
        data = pd.read_csv(tree_path)
        train_size = data.shape[0]*4//5
        train = data.iloc[:train_size,]
        val = data.iloc[train_size:,]
        train.to_csv(save_dir/"train"/tree_subdir.name/"hierarchy.csv", index=False)
        val.to_csv(save_dir/"val"/tree_subdir.name/"hierarchy.csv", index=False)

