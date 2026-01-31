import pandas as pd
import os

# Configuration
INPUT_FILE = 'WSN-DS.csv'
OUTPUT_DIR = 'generated_attack_files'
REQUIRED_COLUMNS = 18

# Column Mapping (Source -> Target)
# Based on WSN-DS.csv header vs Blackhole_cooja.csv header
COLUMN_MAPPING = {
    ' id': 'ID',
    'id': 'ID',
    'who CH': 'Who CH',
    'who_CH': 'Who CH',
    'Expaned Energy': 'Expaned Energy',
    'Attack type': 'Labels' # Just for internal filtering
}

def clean_column_name(name):
    return name.strip()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Clean source columns
    df.columns = [clean_column_name(c) for c in df.columns]
    print(f"Source Columns: {list(df.columns)}")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Attack Types Mapping
    # The dataset uses a mix of numeric and string labels, we need to normalize them.
    # From app.py: 0:Normal, 1:Grayhole, 2:Blackhole, 3:TDMA, 4:Flooding
    attack_map = {
        0: 'Normal', '0': 'Normal', 'Normal': 'Normal',
        1: 'Grayhole', '1': 'Grayhole', 'Grayhole': 'Grayhole',
        2: 'Blackhole', '2': 'Blackhole', 'Blackhole': 'Blackhole',
        3: 'TDMA', '3': 'TDMA', 'TDMA': 'TDMA',
        4: 'Flooding', '4': 'Flooding', 'Flooding': 'Flooding'
    }

    # Normalize Attack Type column
    # Finding the attack type column (usually the last one)
    attack_col = 'Attack type'
    if attack_col not in df.columns:
        # Fallback if name is different
        attack_col = df.columns[-1]
    
    print(f"Using '{attack_col}' as attack label column.")
    
    # Create a normalized label column
    df['normalized_attack'] = df[attack_col].map(attack_map)
    
    # Get unique attacks
    unique_attacks = df['normalized_attack'].dropna().unique()
    print(f"Found attack types: {unique_attacks}")

    # Process each attack type
    for attack_name in unique_attacks:
        print(f"Processing {attack_name}...")
        
        # Filter data
        attack_df = df[df['normalized_attack'] == attack_name].copy()
        
        # Select first 18 columns (Features only)
        # Note: The original dataset has 19 columns (18 features + 1 label)
        # We want only the 18 features for the "Cooja" simulation format
        feature_df = attack_df.iloc[:, :REQUIRED_COLUMNS]
        
        # Rename columns to match Cooja format
        # specific fixes based on our observation
        new_columns = []
        for col in feature_df.columns:
            if col == 'id':
                new_columns.append('ID')
            elif col == 'who CH':
                new_columns.append('Who CH')
            else:
                new_columns.append(col)
        feature_df.columns = new_columns
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, f"{attack_name}.csv")
        feature_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} with shape {feature_df.shape}")

    print("\nGeneration Complete!")
    print(f"Files saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
