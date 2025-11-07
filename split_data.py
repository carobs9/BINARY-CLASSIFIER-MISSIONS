import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/classified_missions_gpt4omini.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['label'],
    random_state=42
)

train_df.to_csv("train_test_datasets/train.csv", index=False)
test_df.to_csv("train_test_datasets/test.csv", index=False)

majority = train_df[train_df.label == 0]
minority = train_df[train_df.label == 1]

# oversample minority to match majority size
minority_upsampled = minority.sample(
    n=len(majority),
    replace=True,
    random_state=42
)

train_balanced = pd.concat([majority, minority_upsampled], ignore_index=True)
train_balanced = train_balanced.sample(frac=1, random_state=42)  # shuffle

train_balanced.to_csv("train_balanced.csv", index=False)