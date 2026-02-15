import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, '..', 'dataset', 'train.csv')
viz_path = os.path.join(base_dir, '..', 'dataset', 'test_sample.csv')
split_train_path = os.path.join(base_dir, '..', 'dataset', 'test_original.csv')

df = pd.read_csv(train_path)
print(f"Total rows in train.csv: {len(df)}")

viz_df = df.sample(n=5000, random_state=42)
remaining_df = df.drop(viz_df.index)

viz_df.to_csv(viz_path, index=False)
remaining_df.to_csv(split_train_path, index=False)