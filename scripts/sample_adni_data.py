import numpy as np
import pandas as pd

RESULT_PATH = "/home/alifim/keymorph/dataset/pair_dice_scores.xlsx"
CSV_PAIR_PATH = "/home/alifim/keymorph/dataset/adni_registration_pairs.csv"

def main():
    # Result of test set
    df_result = pd.read_excel(RESULT_PATH) # ID: Pair Index column
    
    # Merge the two dataframes on the "Pair Index" column
    df_pairs = pd.read_csv(CSV_PAIR_PATH) # ID: based on row number
    # cast "train" column to boolean
    df_pairs["train"] = df_pairs["train"].astype(bool)
    
    # print percentage of train-test split based on "train" columns
    train_count = df_pairs[df_pairs["train"] == True].shape[0]
    test_count = df_pairs[df_pairs["train"] == False].shape[0]
    total_count = train_count + test_count
    train_percentage = (train_count / total_count) * 100
    test_percentage = (test_count / total_count) * 100
    print(f"Train percentage: {train_percentage:.2f}%")
    print(f"Test percentage: {test_percentage:.2f}%")

    # Get 10% of train set randomly and 10% of test set based on lowest Dice scores
    train_set = df_pairs[df_pairs["train"] == True]
    train_set = train_set.reset_index(drop=True) # reset index to ensure it starts from 0
    train_set["Pair Index"] = train_set.index # add pair index for merging
    lowest_10_percent_train = train_set.sample(frac=0.1, random_state=42)
    
    # add pair index to test set for merging
    test_set = df_pairs[df_pairs["train"] == False].reset_index(drop=True) # reset index to ensure it starts from 0
    test_set["Pair Index"] = test_set.index # add pair index for merging
    df_test_merged = test_set.merge(df_result, on="Pair Index")
    lowest_10_percent_test = df_test_merged.nsmallest(int(0.1 * len(test_set)), "Dice Score (tps_10)") 
    lowest_10_percent_test = lowest_10_percent_test[test_set.columns] # keep only the original pair columns
    lowest_10_percent = pd.concat([lowest_10_percent_train, lowest_10_percent_test], ignore_index=True)

    # print the percentage of train and test pairs in the lowest 10% dataset
    lowest_train_count = lowest_10_percent[lowest_10_percent["train"] == True].shape[0]
    lowest_test_count = lowest_10_percent[lowest_10_percent["train"] == False].shape[0]
    lowest_total_count = lowest_train_count + lowest_test_count
    lowest_train_percentage = (lowest_train_count / lowest_total_count) * 100
    lowest_test_percentage = (lowest_test_count / lowest_total_count) * 100
    print(f"10% Train percentage: {lowest_train_percentage:.2f}%")
    print(f"10% Test percentage: {lowest_test_percentage:.2f}%")

    # Save the pairs with the lowest 10% Dice scores to a new CSV file
    # use the original pair columns
    lowest_10_percent.to_csv("/home/alifim/keymorph/dataset/adni_lowest_10_percent_pairs.csv", index=False)

if __name__ == "__main__":
    main()
