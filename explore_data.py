import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/spam.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset info:")
print(df.info())

print("\nClass distribution:")
print(df["label"].value_counts())

df["message_length"] = df["message"].apply(len)

print("\nMessage length statistics:")
print(df.groupby("label")["message_length"].describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nEmpty messages:")
print((df["message"].str.strip() == "").sum())
