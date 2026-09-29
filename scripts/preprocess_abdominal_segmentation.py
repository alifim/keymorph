import pandas as pd
import nibabel as nib
import numpy as np
import os
import shutil

def main(
    csv_path="/midtier/sablab/scratch/alm4065/keymorph/dataset/new_data_pairs.csv",
    output_csv_path="/midtier/sablab/scratch/alm4065/keymorph/dataset/new_data_pairs_preprocessed.csv",
    old_dir="/midtier/sablab/scratch/mch4003",
    new_dir="/midtier/sablab/scratch/alm4065/keymorph/dataset/abdominal_mri",
):
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Track processed files so we don't copy/process the same file twice
    processed_files = set()
    
    path_columns = [
        'fixed_img_path', 'moving_img_path',
        'fixed_seg_path', 'moving_seg_path',
        'fixed_mask_path', 'moving_mask_path',
    ]
    for col in path_columns:
        if col in df.columns:
            df[col] = df[col].astype(object)

    total_rows = len(df)
    print(f"Starting file migration and preprocessing for {total_rows} pairs...")
    print("This may take a few minutes depending on file sizes...\n")
    
    for index, row in df.iterrows():
        # Print progress every 10 rows
        if index % 10 == 0:
            print(f"Processing pair {index} of {total_rows}...")
            
        for col in path_columns:
            if col not in df.columns:
                continue
                
            old_path = row[col]
            
            # ---------------------------------------------------------
            # FIX: Ensure missing paths are filled with the string "None"
            # ---------------------------------------------------------
            if pd.isna(old_path) or str(old_path).strip().lower() in ['none', 'nan', '']:
                df.at[index, col] = "None"  # Write literal string "None"
                continue
            # ---------------------------------------------------------
                
            old_path = str(old_path)
            
            # If the path doesn't belong to the old directory, skip it
            if not old_path.startswith(old_dir):
                continue
                
            # Determine what the base path SHOULD be in your new directory
            new_path = old_path.replace(old_dir, new_dir)
            dir_name = os.path.dirname(new_path)
            
            # ---------------------------------------------------------
            # 1. Handle Segmentation Files (Preprocess and Save)
            # ---------------------------------------------------------
            if 'seg' in col:
                base_name = os.path.basename(new_path)
                if base_name.endswith('.nii.gz'):
                    name = base_name[:-7]
                    ext = '.nii.gz'
                else:
                    name, ext = os.path.splitext(base_name)
                
                final_new_path = os.path.join(dir_name, f"{name}_preprocessed{ext}")
                
                if final_new_path not in processed_files:
                    os.makedirs(dir_name, exist_ok=True) # Create folders if they don't exist
                    
                    if not os.path.exists(final_new_path):
                        img = nib.load(old_path)
                        data = np.round(img.get_fdata())
                        data[(data < 0) | (data > 4)] = 0
                        new_img = nib.Nifti1Image(data.astype(np.uint8), affine=img.affine, header=img.header)
                        nib.save(new_img, final_new_path)
                    if not os.path.isfile(final_new_path):
                        raise FileNotFoundError(f"Segmentation output missing: {final_new_path}")
                    processed_files.add(final_new_path)
                    
                # Update CSV cell
                df.at[index, col] = final_new_path
                
            # ---------------------------------------------------------
            # 2. Handle Image and Mask Files (Just Copy)
            # ---------------------------------------------------------
            else:
                final_new_path = new_path
                
                if final_new_path not in processed_files:
                    os.makedirs(dir_name, exist_ok=True) # Create folders if they don't exist
                    
                    if not os.path.exists(final_new_path):
                        shutil.copy2(old_path, final_new_path)
                    if not os.path.isfile(final_new_path):
                        raise FileNotFoundError(f"Image or mask output missing: {final_new_path}")
                    processed_files.add(final_new_path)
                    
                # Update CSV cell
                df.at[index, col] = final_new_path

    # Save the final, fully updated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nDone! All files migrated and preprocessed.")
    print(f"Updated CSV saved to: {output_csv_path}")

if __name__ == "__main__":
    main()