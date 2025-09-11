import pandas as pd
import re

def parse_fasta_file(file_path, label):
    """
    Parse a FASTA-like file and extract gene names, promoter/non-promoter labels, and sequences.
    
    Args:
        file_path (str): Path to the input .txt file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Gene_Name', 'Label', 'Sequence']
    """
    data = []
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split by '>' to get individual entries
    entries = content.split('>')[1:]  # Skip first empty element
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 2:
            continue
            
        # Parse header line
        header = lines[0]
        sequence = ''.join(lines[1:])  # Join all sequence lines
        
        # Extract gene name (first part before space)
        gene_name = header.split()[3]

        # Remove ";" from gene name if present
        gene_name = gene_name.replace(";", "")
        
        # label = 'Non-Promoter'
        
        data.append({
            'Gene_Name': gene_name,
            'Label': label,
            'Sequence': sequence
        })
    
    return pd.DataFrame(data)

def save_to_csv(df, output_path):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path for output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

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
    # ===================STEP 1===================
    # Parse the input file
    input_file = "./dataset/NonPromoterSequence.txt"
    df = parse_fasta_file(input_file, "Non-Promoter")
    # Save to CSV
    output_file = "./dataset/nonpromoter_sequences.csv"
    save_to_csv(df, output_file)

    input_file = "./dataset/PromoterSequence.txt"
    df = parse_fasta_file(input_file, "Promoter")
    # Save to CSV
    output_file = "./dataset/promoter_sequences.csv"
    save_to_csv(df, output_file)

    # ===================STEP 2===================
    # Replace with your actual file paths
    # file1 = "promoter_sequences.csv"
    # file2 = "nonpromoter_sequences.csv"
    # output = "dataset.csv"
    
    # merge_csv_files(file1, file2, output)

    # Display first few rows
    # print(df.head())