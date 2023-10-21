"""This module contains code for model predictions """
import numpy as np
import pandas as pd
import torch
from load_dataset import DataLoad
from model import ModelBert
from validation import Validator


class Predictor:
    """
    A class for predicting the correct long form for a given acronym
    in a context paragraph.

    """
    def __init__(self, config, device: str = "cuda"):
        model = ModelBert.from_dict(config)
        self.device = torch.device(device)
        self.loaded_model = model.load_model()
        self.config = config

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

    def process_data_pred(self, sentence, full_forms):
        """
        processes the given context sentence into tokenized for
        by modifying in format cls token + full_form+sep token+ context+sep token
        Args: sentence (string)
              full_forms (list): containing possible long forms
        Returns:
            data_frame (pandas.DataFrame): with columns sentence and label
        """
        sentences = {"sentence": [], "label": []}
        for i in full_forms:
            text = (
                self.config["cls_token"]
                + i
                + " "
                + self.config["separator_token"]
                + " "
                + sentence
                + " "
                + self.config["separator_token"]
                + " "
            )
            sentences["sentence"].append(text)
            sentences["label"].append(0)
        data_frame = pd.DataFrame(sentences, columns=["sentence", "label"])
        return data_frame

    def correct_fullform(self, predictions, full_form):
        """
        Predicts the right full form for the acronym in the given context
        from the list of full forms
        Args:

        predictions: A list of predictions returned by the model.
        full_form: A list of full forms of the acronyms present in the input sentences.

        Returns:
        A tuple of the predicted full form and the confidence score.
        """
        predictions_flat = [i for a in predictions for i in a]
        confidence_score = [
            value for array in predictions_flat for value in array.flatten()
        ]
        # Convert the list to a numpy array
        confidence_scores = np.array(confidence_score)
        # Get the index of the maximum value
        max_index = np.argmax(confidence_scores)
        return full_form[max_index], confidence_scores[max_index]

    def predict(self, context, full_forms):
        """
        based on config file, given context and full_forms, predicts the
        right full form along with confidence score
        Args:
            config file:json file
            context_sample: context sentence in which the acronym is used
            full_forms: list of possible full_forms for that context
        returns:  correct long form along with its confidence score
        """
        data_frame = self.process_data_pred(context, full_forms)
        loader = DataLoad.from_dict(self.config)
        dataloader = loader.load_pred_data(data_frame)
        validator = Validator.from_dict(self.config)
        _, _, predictions, _ = validator.loop_dataloader(
            dataloader, mode="pred", model=self.loaded_model
        )
        fullform, score = self.correct_fullform(predictions, full_forms)
        return fullform, score
