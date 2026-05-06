import json
import os
import numpy as np
import csv
import sys
import argparse

def main():
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(description="Process JSON metrics and export to CSV.")
    parser.add_argument(
        "json_filepath", 
        nargs="?", # Makes it optional in case you want a default
        help="Path to the summary_unimodal.json file"
    )
    
    args = parser.parse_args()
    json_filepath = args.json_filepath
    folder_path = os.path.dirname(json_filepath)

    # 2. Attempt to open the file
    try:
        with open(json_filepath, 'r') as f:
            data = json.loads(f.read())
    except FileNotFoundError:
        print(f"Error: Could not find the file at {json_filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    results = {}
    found_transforms = set()

    # Process the JSON data
    for key, values in data.items():
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        
        parts = key.split(':')
        metric = parts[0]
        transform = parts[-1]
        
        found_transforms.add(transform)
        
        if transform not in results:
            results[transform] = {}
        results[transform][metric] = {'mean': mean, 'std': std}

    # Sort transforms: affine first, then TPS sorted dynamically by lambda value
    transforms = sorted(
        list(found_transforms), 
        key=lambda x: -1 if x == 'affine' else float(x.split('_')[1])
    )

    # Clean labels
    labels = {
        "affine": "Affine (Baseline)",
        "tps_0": "TPS (λ = 0)",
        "tps_0.01": "TPS (λ = 10⁻²)",
        "tps_0.1": "TPS (λ = 0.1)",
        "tps_1": "TPS (λ = 1)",
        "tps_10": "TPS (λ = 10)",
        "tps_100": "TPS (λ = 100)"
    }

    metrics = ["harddice", "softdice", "ssim", "mse", "hausd"]
    headers = ["Transformation", "Hard Dice (↑)", "Soft Dice (↑)", "SSIM (↑)", "MSE (↓)", "Hausdorff (↓)"]

    # Write the data directly to a CSV file
    output_file = os.path.join(folder_path, "slides_table.csv")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for t in transforms:
            label = labels.get(t, t) 
            row = [label]
            
            for m in metrics:
                if m in results[t]:
                    mean = results[t][m]['mean']
                    std = results[t][m]['std']
                    
                    # Format decimals
                    if m == 'hausd':
                        s = f"{mean:.2f} ± {std:.2f}"
                    else:
                        s = f"{mean:.3f} ± {std:.3f}"
                else:
                    s = "N/A"
                    
                row.append(s)
                
            writer.writerow(row)

    print(f"Success! CSV saved to: {output_file}")

if __name__ == "__main__":
    main()