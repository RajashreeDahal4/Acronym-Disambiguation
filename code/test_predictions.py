"""This module contains code for predictions of test dataset"""
import json
import numpy as np
import pandas as pd
import wandb


def process_test_data(config, test_data):
    """
    Processes test data for the model
    Args: test_data: A dictionary of  6keys containing sentence, labels,
    acronym,full_form,correct_full_form which are lists
    Returns:
        dataframe: a pandas dataframe with sorted rows on context level
        with columns=["acronym","full_form","correct_full_form","true_label"]
    """

    sentences = {
        "acronym": test_data["test_acro"],
        "full_form": test_data["test_full_form"],
        "correct_full_form": test_data["test_correct_form"],
        "sentence": test_data["test_sent"],
        "true_label": test_data["test_labels"],
    }
    data_frame = pd.DataFrame(
        sentences,
        columns=[
            "acronym",
            "full_form",
            "correct_full_form",
            "sentence",
            "true_label",
        ],
    )
    data_frame["sort_col"] = data_frame["sentence"].apply(
        lambda x: x.split(config["separator_token"])[1]
    )
    # Sort the DataFrame by the new column
    data_frame = data_frame.sort_values(
        ["sort_col", "acronym", "full_form"], ascending=[True, True, True]
    )
    # Drop the temporary column
    data_frame = data_frame.drop(columns=["sort_col"])
    data_frame = data_frame.drop_duplicates(
        subset=[
            "acronym",
            "full_form",
            "correct_full_form",
            "sentence",
            "true_label",
        ]
    )
    return data_frame


def reduce_group(group, name):
    """
    takes a group from pandas dataframe and reduces it to a dictionary with selected
    values for the group
    Args:
        'group' (pandas.DataFrame) : A group of rows from a pandas DataFrame
        'name'(str): the name of the group
    Returns: A dictionary with context,full_forms,
    correct_full_form,predicted_full_form and confidence_score as keys

    """
    # filter the group to only include rows where "predicted value" is 1
    filtered_group = group[group["predictions"] == 1]

    if len(filtered_group) == 0:
        return {}
    # return a dictionary with the reduced values for the group
    full_forms = list(group["full_forms"])
    correct_full_form = list(group["correct_full_form"])
    confidence_score = list(filtered_group["confidence_score"])
    predicted_full_form = list(filtered_group["full_forms"])
    max_len = max(
        len(full_forms),
        len(correct_full_form),
        len(confidence_score),
        len(predicted_full_form),
    )

    # Pad the shorter arrays with zeros to match the length of the longest array
    full_forms = [
        full_forms[i] if i < len(full_forms) else "None" for i in range(max_len)
    ]
    correct_full_form = [
        correct_full_form[i] if i < len(correct_full_form) else "None"
        for i in range(max_len)
    ]
    confidence_score = [
        confidence_score[i]
        if i < len(confidence_score)
        else np.zeros(1, dtype=np.float32)
        for i in range(max_len)
    ]
    predicted_full_form = [
        predicted_full_form[i] if i < len(predicted_full_form) else "None"
        for i in range(max_len)
    ]
    return {
        "context": name,
        "full_forms": full_forms,
        "correct_full_form": correct_full_form,
        "predicted_full_form": predicted_full_form,
        "confidence_scores": confidence_score,
    }


def prediction_output(data_frame):
    """
    creates a summary table of the model's predictions for each context
    in the input dataframe
    Args: data_frame(pandas.DataFrame): A dataframe containing predictions made
            by the model on test data
    Returns: None
    """
    # group the DataFrame by the "context" column
    grouped = data_frame.groupby(["sort_col", "correct_full_form"])
    reduced_df = pd.concat(
        [pd.DataFrame(reduce_group(group, name[0])) for name, group in grouped],
        ignore_index=True,
    )
    reduced_df["confidence_scores"] = [
        score[0] for score in reduced_df["confidence_scores"]
    ]

    reduced_df = reduced_df.reset_index(drop=True)
    reduced_df["confidence_scores"] = reduced_df["confidence_scores"].apply(
        lambda lst: json.dumps(lst)
    )
    grouped = (
        reduced_df.groupby(["context", "correct_full_form"])
        .agg(
            {
                "full_forms": lambda x: [i for i in x if i != "None"],
                "predicted_full_form": lambda x: [i for i in x if i != "None"],
                "confidence_scores": lambda x: [i for i in x if i != "0.0"],
            }
        )
        .reset_index()
    )
    new_order = [
        "context",
        "full_forms",
        "correct_full_form",
        "predicted_full_form",
        "confidence_scores",
    ]
    table = grouped.reindex(columns=new_order)
    table = pd.DataFrame(data=table)
    remove_quotes = lambda x: x.replace('"', "")
    table["context"] = table["context"].apply(remove_quotes)
    # Log data in wandb as a dictionary
    table = wandb.Table(dataframe=table)
    # Log the table to the run
    wandb.log({"predictions": table})
