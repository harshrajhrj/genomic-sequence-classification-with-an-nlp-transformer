import pandas as pd
import numpy as np

def merge_csv_files(file1_path, file2_path, output_path):
    """
    Merge two CSV files and reorder rows in random order.
    
    Args:
        file1_path (str): Path to first CSV file
        file2_path (str): Path to second CSV file
        output_path (str): Path for output merged CSV file
    """
    # Read the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Merge the dataframes (concatenate vertically)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Shuffle rows randomly
    shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to output file
    shuffled_df.to_csv(output_path, index=False)
    
    print(f"Merged {len(df1)} + {len(df2)} = {len(shuffled_df)} rows")
    print(f"Output saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file1 = "promoter_sequences.csv"
    file2 = "nonpromoter_sequences.csv"
    output = "dataset.csv"
    
    merge_csv_files(file1, file2, output)