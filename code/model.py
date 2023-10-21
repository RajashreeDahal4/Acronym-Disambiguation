"""This module contains code for model architecture."""
import importlib
import os
import pandas as pd
import torch
from transformers.utils import logging
from validation import Validator
import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.set_verbosity(40)


class ModelBert:
    """
    A class for predicting the correct long form for a given acronym
    in a context paragraph using a pre-trained BERT model and tokenizer.

    """

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        self.state_dict = None
        self.data_frame = None
        self.model = None
        self.full_form = None
        self.tokenizer = None

    @classmethod
    def from_dict(cls, cfg):
        """
        Creates an instance of ClassName from a dictionary.
        Parameters:
        ----------
        cfg: dict
            A dictionary containing configuration information for the instance.

        Returns:
        -------
        ClassName
            An instance of the ClassName with the given configuration.
        """
        return cls(cfg, device=cfg.get("device"))

    def make_model(self):
        """
        Instantiates a pre-trained BERT model and tokenizer.
        Returns:
            A tuple containing the model and the tokenizer.
        """
        # Dynamically import the transformers module
        module_name = self.config["module_name"]
        transformers = importlib.import_module(module_name)
        # Dynamically get the BertForSequenceClassification class from the transformers module
        model_class = getattr(transformers, self.config["model"])
        model = model_class.from_pretrained(
            self.config["model_type"], num_labels=self.config["num_labels"]
        ).to(self.device)

        token_class = getattr(transformers, self.config["tokenizer"])

        tokenizer = token_class.from_pretrained(
            self.config["model_type"],
            is_split_into_words=True,
            additional_special_tokens=self.config["additional_special_tokens"],
        )
        model.resize_token_embeddings(len(tokenizer))
        # set the pad token of the model's configuration
        if self.config["model_type"] == "gpt2":
            model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer

    def load_model(self):
        """loads the saved  model in evaluation mode"""

        self.state_dict = torch.load(
            self.config["saved_model_path"], map_location=self.device
        )
        self.model, self.tokenizer = self.make_model()
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        return self.model

    def test_predict(self, dataloader, data_frame):
        """
        Make predictions on the test set using a trained model and return a pandas
        DataFrame with the predicted labels and confidence scores for each acronym
        and full form pair in the test set.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader object containing the test
                                    set as batches.
            data_frame (pd.DataFrame): A pandas dataframe with columns "acronym","full_form",
             "correct_full_form", "sentence", "true_label",
        Returns:
            a tuple of pandas dataframe, and two lists where
            A pandas DataFrame has columns for the acronym, context sentence, full form,
            correct full form, predicted label, and confidence score for each acronym
            and full form pair in the test set.
            and the two lists are predicted labels and actual labels
        """
        model = self.load_model()
        validator = Validator.from_dict(self.config)
        accuracy, nb_steps, preds, labels = validator.loop_dataloader(
            dataloader, mode="test", model=model
        )
        wandb.log({" Test Accuracy": (accuracy / nb_steps)})
        predictions_flat = [i for a in preds for i in a]
        confidence_score_flat = predictions_flat
        predictions_flat = [
            1 if i >= self.config["test_confidence_threshold"] else 0
            for i in predictions_flat
        ]
        pred = pd.DataFrame(
            {
                "acronym": data_frame.acronym.values,
                "context": data_frame.sentence.values,
                "full_forms": data_frame.full_form.values,
                "correct_full_form": data_frame.correct_full_form.values,
                "predictions": predictions_flat,
                "confidence_score": confidence_score_flat,
            },
            columns=[
                "acronym",
                "context",
                "full_forms",
                "correct_full_form",
                "predictions",
                "confidence_score",
            ],
        )
        pred["sort_col"] = pred["context"].apply(
            lambda x: x.split(self.config["separator_token"])[1]
        )

        return pred, preds, labels
