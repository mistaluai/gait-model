import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_preprocessor import load_gait_sequences


def run_gait_eda(df: pd.DataFrame, dataset_name: str):
    """
    Perform EDA on gait sequence DataFrame.

    Args:
        df (pd.DataFrame): Dataset containing ['sequence', 'label', 'subject', 'angle', 'trial', 'source']
        dataset_name (str): Name of the dataset (for plot titles)
    """
    print(f"=== EDA for {dataset_name} ===\n")
    print("=== Basic Info ===")
    print(df.info())
    print("\n=== Sample Rows ===")
    print(df.head())

    # Convert relevant columns to string to avoid dtype issues
    for col in ['label', 'angle', 'source', 'subject', 'trial']:
        df[col] = df[col].astype(str)

    # 1. Class Distribution
    plt.figure(figsize=(8, 4))
    plot = sns.countplot(data=df, x='label', hue='label',
                         order=df['label'].value_counts().index, palette='tab10')
    handles, labels = plot.get_legend_handles_labels()
    plt.legend(handles, labels, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Class Distribution – {dataset_name}")
    plt.xlabel("Gait Class")
    plt.ylabel("Number of Sequences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Angle Distribution
    plt.figure(figsize=(6, 4))
    plot = sns.countplot(data=df, x='angle', hue='angle',
                         order=df['angle'].value_counts().index, palette='Set2')
    handles, labels = plot.get_legend_handles_labels()
    plt.legend(handles, labels, title="Angle", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Camera Angle Distribution – {dataset_name}")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Number of Sequences")
    plt.tight_layout()
    plt.show()

    # 3. Source Distribution
    plt.figure(figsize=(6, 4))
    plot = sns.countplot(data=df, x='source', hue='source',
                         order=df['source'].value_counts().index, palette='Set3')
    handles, labels = plot.get_legend_handles_labels()
    plt.legend(handles, labels, title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Subfolder (Source) Distribution – {dataset_name}")
    plt.xlabel("Source Type")
    plt.ylabel("Number of Sequences")
    plt.tight_layout()
    plt.show()

    # 4. Sequence Length Stats
    df['seq_len'] = df['sequence'].apply(len)
    print("\n=== Sequence Length Stats ===")
    print(df['seq_len'].describe())

    plt.figure(figsize=(8, 4))
    sns.histplot(df['seq_len'], bins=20, kde=True, color='skyblue')
    plt.title(f"Sequence Length Distribution – {dataset_name}")
    plt.xlabel("Number of Frames")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = "./gei_maps/Multiclass6"
    dataset_name = dataset_path.strip("/").split("/")[-1]
    df = load_gait_sequences(dataset_path, load_images=False)
    run_gait_eda(df, dataset_name)