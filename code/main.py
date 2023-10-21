"""This module contains code for parsing and manipulating JSON data."""
import argparse
import json
from transformers.utils import logging


from preprocessing import DataPreprocess as process
from train import Trainer
from metrics_evaluation import Metrics
from load_dataset import DataLoad
from model import ModelBert
from test_predictions import process_test_data, prediction_output
from predictions import Predictor
import wandb

logging.set_verbosity(40)


def load_train_dataset(config):
    """processes the datasets and loads in dataloader
     Args:
      config file: json
    Returns: train dataloader and validation dataloader
    """
    wandb.init(project="acronym_disambiguation6", config=config)
    train_data, test_data, valid_data = process.from_dict(config).train_format()
    loader = DataLoad.from_dict(config)
    input_ids, labels, attention_masks = loader.token(train_data)
    train_dataloader = loader.load_dataset(
        input_ids, labels, attention_masks, mode="train"
    )
    input_ids, labels, attention_masks = loader.token(valid_data)
    validation_dataloader = loader.load_dataset(
        input_ids, labels, attention_masks, mode="validation"
    )
    test_data = {
        "test_sent": test_data["sentence"],
        "test_labels": test_data["label"],
        "test_acro": test_data["acronym"],
        "test_full_form": test_data["value"],
        "test_correct_form": test_data["correct_full_form"],
    }
    return (train_dataloader, validation_dataloader, test_data)


def test_predictions(config, data):
    """
    predictions of test data and wandb dumps
    Arg(s):
            config : loaded json file
            data:  A dictionary of  6 keys containing sentence, labels, acronym,
            full_form,correct_full_form which are in the form oflists
    """
    # starting predictions
    loader = DataLoad.from_dict(config)
    data_frame = process_test_data(config, data)
    prediction_dataloader = loader.load_pred_data(data_frame)
    model = ModelBert.from_dict(config)
    predict_dataframe, preds, labels = model.test_predict(
        prediction_dataloader, data_frame
    )
    prediction_output(predict_dataframe)
    Metrics.from_dict(config).print_metrics(
        Metrics.from_dict(config).metric_fn(preds, labels), mode="test"
    )
    Metrics.from_dict(config).roc_auc_confusion_matrix(preds, labels)


def train(config_file):
    """
    This function trains the  model and the
    required loss and metrics are dumped in wandb
    Args:
        config_file: json file
    """
    with open(config_file, encoding="utf-8") as file_data:
        config = json.load(file_data)
    (
        train_dataloader,
        validation_dataloader,
        test_data,
    ) = load_train_dataset(config)
    preds, labels = Trainer.from_dict(config).model_train(
        train_dataloader, validation_dataloader
    )
    Metrics.from_dict(config).print_metrics(
        Metrics.from_dict(config).metric_fn(preds, labels), mode=""
    )

    test_predictions(config, test_data)
    wandb.finish()


def predicts(config_file, context, full_forms):
    """
    Predicts the possible full_forms and their confidence_scores which exceed the
    confidence_threshold given context and full_forms as input
    Arg(s):
            config_file: json file for config
            context(string): a sentence containing short form
            full_forms(list): list of possible full form for the given
            short form in the given context
    """
    with open(config_file, encoding="utf-8") as files:
        config = json.load(files)
    predictor = Predictor.from_dict(config)
    result = predictor.predict(context, full_forms)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training \
                and prediction with given configuration file."
    )
    subparsers = parser.add_subparsers(dest="subparser_name", help="sub-command help")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config_file", type=str, help="Path of the configuration file."
    )

    pred_parser = subparsers.add_parser(
        "predicts", help="Make predictions using the model"
    )
    pred_parser.add_argument(
        "--config_file", type=str, help="Path of the configuration file."
    )
    pred_parser.add_argument("--a", type=str, help="context where the acronym is used")
    pred_parser.add_argument("--b", nargs="+", help="list of long forms")

    args = parser.parse_args()

    if args.subparser_name == "train":
        train(args.config_file)
    if args.subparser_name == "predicts":
        predicts(args.config_file, args.a, args.b)
    else:
        parser.print_help()
