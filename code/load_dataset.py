""" Importing modules neccessary for loading the preprocessed dataset"""
import importlib
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class DataLoad:
    """This class loads and tokenizes the data for a text classification task using BERT"""

    def __init__(
        self,
        config,
        batch_size: int = 64,
        model_type: str = "bert-base-cased",
        additional_special_tokens=None,
    ):
        """
        A class for loading and preprocessing data for a text classification task using BERT.

        Attributes:
            tokenizer: A tokenizer for tokenizing input sentences.
            additional_special_tokens: additional special tokens used in the model
            batch_size: The batch size for training and validation data.

        """
        self.config = config
        self.batch_size = batch_size
        if additional_special_tokens is None:
            additional_special_tokens = self.config["additional_special_tokens"]
        module_name = self.config["module_name"]
        transformers = importlib.import_module(module_name)
        # Dynamically get the BertForSequenceClassification class from the transformers module
        token_class = getattr(transformers, self.config["tokenizer"])
        self.tokenizer = token_class.from_pretrained(
            model_type,
            is_split_into_words=True,
            additional_special_tokens=self.config["additional_special_tokens"],
        )
        self.data_frame = None

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Creates an instance of ClassName from a dictionary.
        Parameters:
        ----------
        cfg: dict
            A dictionary containing configuration information for the instance.
        additional_special_tokens: additional special tokens used in the model

        Returns:
        -------
        ClassName
            An instance of the ClassName with the given configuration."""
        return cls(
            cfg,
            batch_size=cfg.get("batch_size"),
            model_type=cfg.get("model_type"),
            additional_special_tokens=cfg.get("additional_special_tokens"),
        )

    def token(self, data):
        """
        Tokenize input sentences and create attention masks.
        Args:
            data: Pandas dataframe

        Returns:
            A tuple of three lists: the tokenized form of each input sentence,
            the corresponding labels, and the attention masks for each sentence.
        """
        sentence = list(data["sentence"])
        labels = list(data["label"])
        # Load the pre-trained BERT model
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentence]
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(
            input_ids,
            maxlen=self.config["pad_len"],
            dtype="long",
            truncating="post",
            padding="post",
        )
        # Create attention masks
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return input_ids, labels, attention_masks

    def load_dataset(self, inputs, labels, masks, mode):
        """Load the dataset in dataloader
        Args:
            inputs: list of tokenized form of each input sentence
            labels: list of labels
            masks: list of attention masks for each sentence
        Returns:
            Pytorch dataloader
        """
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        masks = torch.tensor(masks)
        data = TensorDataset(inputs, masks, labels)
        if mode in ["test", "validation"]:
            sampler = SequentialSampler(data)
        else:
            sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        return dataloader

    def load_pred_data(self, data_frame):
        """
        loads the pred_data
        Args: dataframe: a pandas dataframe
        Returns: a pytorch dataloader for prediction dataset
        """
        if "true_label" in data_frame.columns:
            labels = list(data_frame["true_label"])
        else:
            labels = [0] * len(data_frame)
        data_frame["label"] = labels
        input_ids, labels, attention_masks = self.token(data_frame)
        prediction_dataloader = self.load_dataset(
            input_ids, labels, attention_masks, mode="test"
        )
        return prediction_dataloader
