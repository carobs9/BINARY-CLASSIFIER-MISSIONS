import pandas as pd
from sklearn.model_selection import train_test_split

DATA_OF_CHOICE = 'activities'

df = pd.read_csv(f'data/classified_{DATA_OF_CHOICE}_gpt4omini_PROMPT2.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['label'],
    random_state=42
)

train_df.to_csv(f"train_test_datasets/train_{DATA_OF_CHOICE}_PROMPT2.csv", index=False)
test_df.to_csv(f"train_test_datasets/test_{DATA_OF_CHOICE}_PROMPT2.csv", index=False)

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

train_balanced.to_csv(f"train_test_datasets/train_balanced_{DATA_OF_CHOICE}_PROMPT2.csv", index=False)