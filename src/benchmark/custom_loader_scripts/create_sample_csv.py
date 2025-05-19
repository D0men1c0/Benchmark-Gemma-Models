import pandas as pd
from pathlib import Path
import random

def generate_sample_csv_dataset(output_dir: str = "user_scripts/data",
                                train_samples: int = 100,
                                validation_samples: int = 30,
                                test_samples: int = 20):
    """
    Generates sample CSV files (train, validation, test) for demonstration.

    Each CSV will have 'id', 'text_content', and 'category_label' columns.
    'text_content' will be sample sentences.
    'category_label' will be an integer representing a class (0, 1, or 2).
    'id' will be a unique string identifier for each sample.

    :param output_dir: Directory to save the generated CSV files.
    :param train_samples: Number of training samples to generate.
    :param validation_samples: Number of validation samples to generate.
    :param test_samples: Number of test samples to generate.
    :return: None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_phrases = [
        "The weather is sunny and warm today.",
        "This is a complex problem that requires a creative solution.",
        "Artificial intelligence is rapidly transforming various industries.",
        "Reading books expands your knowledge and vocabulary.",
        "The new movie received excellent reviews from critics.",
        "Learning a new language can be challenging but rewarding.",
        "Healthy eating and regular exercise are important for well-being.",
        "The team worked collaboratively to achieve their goals.",
        "Technology continues to evolve at an unprecedented pace.",
        "Environmental conservation is a global responsibility."
    ]
    categories = [0, 1, 2] # Example: sentiment (neg, neu, pos) or topic categories

    def create_split_data(num_samples, split_name):
        data = []
        for i in range(num_samples):
            text = f"Sample {split_name} text {i+1}: {random.choice(sample_phrases)} {random.choice(sample_phrases)}"
            label = random.choice(categories)
            sample_id = f"{split_name}_{i:04d}"
            data.append({"id": sample_id, "text_content": text, "category_label": label})
        return data

    # Generate data for each split
    train_data = create_split_data(train_samples, "train")
    validation_data = create_split_data(validation_samples, "validation")
    test_data = create_split_data(test_samples, "test")

    # Create DataFrames and save to CSV
    df_train = pd.DataFrame(train_data)
    df_validation = pd.DataFrame(validation_data)
    df_test = pd.DataFrame(test_data)

    train_file = output_path / "sample_train_dataset.csv"
    validation_file = output_path / "sample_validation_dataset.csv"
    test_file = output_path / "sample_test_dataset.csv"

    df_train.to_csv(train_file, index=False)
    df_validation.to_csv(validation_file, index=False)
    df_test.to_csv(test_file, index=False)

    print(f"Sample CSV datasets generated in '{output_path}':")
    print(f" - {train_file.name} ({len(df_train)} samples)")
    print(f" - {validation_file.name} ({len(df_validation)} samples)")
    print(f" - {test_file.name} ({len(df_test)} samples)")

if __name__ == "__main__":
    generate_sample_csv_dataset()