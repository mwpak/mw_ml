"""v1.1 - 250125"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")


def logging(msg, outdir, log_fpath):
    fpath = os.path.join(outdir, log_fpath)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print(msg)


def balance_by_label(
    train_df,
    balance_column="Y",
    balance_values=None,
    random_seed=42,
    min_freq=100,
    visualize=False,
    outdir=None,
):
    """
    Balance the data by oversampling or undersampling based on the specified column.

    Parameters:
    - train_df (pd.DataFrame): The training dataframe.
    - balance_column (str): The column to balance the data on. Default is 'Y'.
    - balance_values (list): The specific values to consider for calculating the reference number. Default is None.
    - random_seed (int): The random seed for reproducibility. Default is 42.
    - min_freq (int): The minimum frequency threshold for upsampling. Default is 100.
    - visualize (bool): Whether to visualize the resulting data label composition. Default is False.
    - outdir (str): The directory to save the visualization. Default is None.

    Returns:
    - pd.DataFrame: The balanced dataframe.
    """
    # Determine the maximum count for balancing
    if balance_values is None:
        max_count = train_df[balance_column].value_counts().max()
    else:
        max_count = (
            train_df[train_df[balance_column].isin(balance_values)][balance_column]
            .value_counts()
            .max()
        )

    # Ensure the max_count is at least min_freq
    if max_count < min_freq:
        max_count = min_freq

    balance_values = train_df[balance_column].unique()

    balanced_dfs = []
    for value in balance_values:
        value_df = train_df[train_df[balance_column] == value]
        # Upsample or downsample to match max_count
        if value_df.shape[0] < max_count:
            value_df = value_df.sample(
                n=max_count, replace=True, random_state=random_seed
            )
        elif value_df.shape[0] > max_count:
            value_df = value_df.sample(
                n=max_count, replace=False, random_state=random_seed
            )
        balanced_dfs.append(value_df)

    balanced_df = pd.concat(balanced_dfs)

    assert balanced_df[balance_column].value_counts().nunique() == 1

    # Visualize the resulting data label composition if requested
    if visualize:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=balanced_df, x=balance_column, order=balance_values)
        plt.title(f"Balanced Data Composition by {balance_column}")
        plt.xlabel(balance_column)
        plt.ylabel("Count")
        if outdir:
            plt.savefig(
                os.path.join(outdir, f"balanced_data_composition_{balance_column}.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    return balanced_df


# * ====== Performance metrics ====== * #
def evaluate_prediction(
    pred_df, pred_col="pred", y_col="Y", labels=[0, 1], pos_label=1, report=True
):
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        f1_score,
        precision_score,
        accuracy_score,
        recall_score,
    )

    # === Report === #
    # Confusion matrix
    conf_matrix = confusion_matrix(pred_df[y_col], pred_df[pred_col], labels=labels)

    # Classification report
    if report:
        print(classification_report(pred_df[y_col], pred_df[pred_col]))

    # === Metrics scores === #
    # TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(
        pred_df[y_col], pred_df[pred_col], labels=labels
    ).ravel()
    if report:
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # Accuracy
    acc_man = (tp + tn) / (tp + tn + fp + fn)
    acc = accuracy_score(pred_df[y_col], pred_df[pred_col])
    assert round(acc, 3) == round(acc_man, 3), f"Accuracy mismatch: {acc} != {acc_man}"

    # Precision
    prec_man = tp / (tp + fp)
    prec = precision_score(
        pred_df[y_col], pred_df[pred_col], labels=labels, pos_label=pos_label
    )
    assert round(prec, 3) == round(prec_man, 3), (
        f"Precision mismatch: {prec} != {prec_man}"
    )

    # Recall
    recall_man = tp / (tp + fn)
    recall = recall_score(
        pred_df[y_col], pred_df[pred_col], labels=labels, pos_label=pos_label
    )
    assert round(recall, 3) == round(recall_man, 3), (
        f"Recall mismatch: {recall} != {recall_man}"
    )

    # F1 score
    f1_man = 2 * (prec * recall) / (prec + recall)
    f1 = f1_score(pred_df[y_col], pred_df[pred_col], labels=labels, pos_label=pos_label)
    assert round(f1, 3) == round(f1_man, 3), f"F1 score mismatch: {f1} != {f1_man}"

    return acc, prec, recall, f1, conf_matrix


# # Example usage
# acc, prec, recall, f1, conf_matrix = evaluate_prediction(
#     target_df,
#     pred_col="best_pred_adj",
#     y_col="Y",
#     labels=["WT", "MUT"],
#     pos_label="MUT",
# )
# print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
