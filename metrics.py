import argparse
import numpy as np
import pandas as pd

#-------------------------------------------------------------------------------

# Compute prediction level
def compute_prediction_level(df):
    """This function choose the lowest level for which the confidence of the model is above the level threshold.
    """

    results = []
    for instance_id, group in df.groupby("instance_id"):
        # Sort levels numerically or lexicographically
        group = group.sort_values("level")

        # Filter levels where confidence >= threshold
        valid_levels = group[group["confidence"] >= group["threshold"]]

        if not valid_levels.empty:
            chosen_level = valid_levels.iloc[0]["level"]  # lowest valid level
        else:
            chosen_level = None  # or np.nan

        results.append({"instance_id": instance_id, "prediction_level": chosen_level})

    # Convert results to DataFrame and merge back
    result_df = pd.DataFrame(results)
    df = df.merge(result_df, on="instance_id", how="left")

    return df

#-------------------------------------------------------------------------------

# Accuracy
def micro_accuracy(df):
    return df['correct'].mean()

# Macro accuracy at each level
def macro_accuracy(df):
    return df['']

# Coverage
def coverage(df):
    """Proportion of instances where the model made any prediction
      (i.e., had confidence ≥ threshold at some level).
    """
    return df['prediction_made'].mean()

# Coverage per level
def coverage_per_level(df):
    return df['prediction_level'].value_counts(dropna=False).sort_index()/len(df)

# Correct @ Level
def correct_at_each_level(df):
    level_accuracy = df.dropna().groupby('prediction_level')['correct'].mean()
    return level_accuracy.to_dict()

# Average Prediction Level
def average_prediction_level(df):
    return df['prediction_level'].mean()

# No Prediction Rate
def no_prediction_rate(df):
    return 1 - coverage(df)

# Mean Confidence of Correct vs Incorrect Predictions
def confidence_stats(df):
    result = {}
    for outcome in [0, 1]:
        ids = df.loc[df['correct'] == outcome, 'instance_id']
        subset = df[df['instance_id'].isin(ids)]
        result[f'mean_confidence_correct_{outcome}'] = subset['confidence'].mean()
    return result

def hierarchical_metric(df, rewards=None, penalties=None):
    if rewards is None:
        # rewards = lambda level: (3-level)/6
        rewards = pd.Series([1/2 - x/6 for x in range(3)]) # [1/2, 1/3, 1/6]
    if penalties is None:
        # penalties = lambda level: (-1-level)/6)
        penalties = pd.Series([-(1+x)/6 for x in range(3)]) # [-1/6, -1/3, -!/2]
    m = np.where(
    df['confidence']>df['threshold'], 
    np.where(
        df['label']==df['prediction'],
        rewards[df['level']], 
        penalties[df['level']]),
    0)
    return sum(m)/df['instance_id'].nunique()

# Run all metrics in one call
def evaluate_all_metrics(df):
    return {
        'micro_accuracy': micro_accuracy(df),
        'coverage': coverage(df),
        'coverage_per_level' : coverage_per_level(df),
        'average_prediction_level': average_prediction_level(df),
        'correct_at_each_level': correct_at_each_level(df),
        **confidence_stats(df),
        "hierarchical_metric":hierarchical_metric(df),
    }

def main(csv="mini_results.csv"):
    df = pd.read_csv(csv)
    df=compute_prediction_level(df)
    print(evaluate_all_metrics(df))

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="mini_results.csv", help="Path to the result files.")
    args = parser.parse_args()
    main(args.file)

if __name__=='__main__':
    cli()