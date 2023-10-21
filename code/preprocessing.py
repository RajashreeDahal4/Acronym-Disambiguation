"""This module contains code for parsing and manipulating JSON data."""
import json
import re
import os
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class DataPreprocess:
    """This class preprocesses the given dataset and returns a preprocessed dataframe"""

    def __init__(self, config, similarity_threshold: float = 0.7, datapath: str = None):
        """
        Constructor to instantiate DataPreprocess class.

        Initializes the following attributes:
        - file_list: A list of file names in datapath
        - json_list: A list of file names ending with '.json' in datapath.
        - data_frame: A pandas DataFrame to store the preprocessed data.
        - diction: A dictionary to store the list of full-form values for each acronym.
        - similarity_threshold: A float number representing the similarity threshold for
        matching acronyms with their full-forms for given context.
        """
        self.config = config
        # Define the similarity threshold
        if datapath is None:
            self.datapath = self.config["datapath"]
        else:
            self.datapath = datapath
        self.vectorizer = TfidfVectorizer()
        self.similarity_threshold = similarity_threshold
        self.diction = {}
        self.data_frame = pd.DataFrame(columns=["acronym", "value", "context"])
        file_list = os.listdir(self.datapath)
        self.json_list = [f for f in file_list if f.endswith(".json")]

    @classmethod
    def from_dict(cls, cfg: dict):
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
        return cls(
            cfg,
            similarity_threshold=cfg.get("similarity_threshold"),
            datapath=cfg.get("datapath"),
        )

    def get_value(self, diction, full_form):
        """
        Returns the value of the given key in the dictionary if the key exists, None otherwise.

        Args:
        - diction: A dictionary.
        - full_form: A key to look up in the dictionary.

        Returns:
        - A value associated with the given key in the dictionary or None.
        """
        for key, value in diction.items():
            if key == full_form:
                return value
        return None

    def create_dataframe_from_data(self):
        """
        Returns a pandas DataFrame containing the preprocessed data.

        This method reads data from '.json' files in 'datapath, preprocesses it,
        and stores it in a pandas DataFrame.
        Returns: data_frame (pandas.DataFrame)
        """
        for file in self.json_list:
            with open(self.datapath + file, encoding="utf-8") as file_data:
                data = json.load(file_data)
            for k, v_d in data.items():
                self.diction[k] = list(v_d.keys())
            data = list(data.items())
            data_frame2 = pd.DataFrame(data, columns=["acronym", "value"])
            # dropping the empty key from the dictionary which maps with totally empty list.
            # Note: this full_form has already been added to the global diction list
            data_frame2["value"] = data_frame2["value"].apply(
                lambda d: {k: v for k, v in d.items() if v}
            )
            data_frame2["context"] = data_frame2["value"]
            data_frame2 = data_frame2.explode("value")
            data_frame2["context"] = data_frame2.apply(
                lambda row: self.get_value(row["context"], row["value"]), axis=1
            )
            mask = data_frame2["context"].apply(lambda x: bool(x))
            data_frame2 = data_frame2[mask]
            data_frame2 = data_frame2.explode("context")
            data_frame2 = data_frame2.reset_index(drop=True)
            self.data_frame = pd.concat([self.data_frame, data_frame2])
        self.data_frame = self.data_frame.reset_index(drop=True)
        self.data_frame = self.data_frame.astype(str)
        # self.data_frame = self.data_frame[1:1000]
        return self.data_frame

    def acro_in_context_with_s(self, acronym, context):
        """
        Returns the context with the given acronym's plural form or singluar form in parenthes is
        removed from it if it is present in the context.
        Parameters:
        acronym (str): The acronym in parenthesis is to be removed from the context.
        context (str): The context to check for the presence of the given acronym's plural form.

        Returns:
        str: The modified context with the given acronym's plural or singular in parenthesis form
        removed if it was present, otherwise the unmodified context.
        """
        result = f"({acronym}(s)?)"
        if result in context:
            context = context.replace(result, "")
        return context

    def full_form_with_article(self, value, context):
        """
        Returns the context with the given full form value preceded by "the", "The", "a", or "A"
        removed from it if it is present in the context.

        Parameters:
        value (str): The full form value whose variants with articles are to be removed from the
        context. context (str): The context to check for the presence of the given full form
        value with any of the four article variants.

        Returns:
        str: The modified context with the given full form value preceded by any of the
        four article variants removed if it was present, otherwise the unmodified context.
        """
        value = " " + value
        context = context.replace(f"The {value}", value).replace(f"the {value}", value)
        context = context.replace(f"A {value}", value).replace(f"a {value}", value)
        return context

    def replace_fullform_context(self, value):
        """
        Args:
            value (str): The input string to be processed.

        Returns:
            str or int: The processed string or 1 if the input string was empty after processing.
        """
        value = re.sub(
            r"[\(\[].*?[\)\]]|<.*?>|{.*?}|\\x80|\\x89|\\x8f|\\x92|\\x93|\\x96|\\x99|\n"
            r"|\\xao|¢|=|§|¨|©|!|\*|®|¯|±|¿|:|-|\+|@|→|⟶|>|{|}|∑|φ|ψ|\^",
            "",
            value,
        )
        if value == "":
            return "1"
        return value

    def full_form_with_s(self, value, context):
        """
        Parameters:
        value (str): the word whose plural form needs to be replaced with its singular form.
        context (str): the string in which the replacement needs to be made.
        Returns:
        The context string with the plural form of value replaced with its singular form.
        """
        return context.replace(f"{value}s", value).replace(f"{value.lower()}s", value)

    def full_form_in_short(self, value, context):
        """
        Args:
            value (str): A string representing the value whose short form
            needs to be replaced with the full form.
            context (str): A string representing the context in which the short
            form needs to be replaced with the full form.
        Returns:
            str: The updated context after replacing the short form of
            the value with the full form.
        """
        return context.replace(" " + value.lower() + " ", value)

    def replace_full_form_with_acronym(self, value, context, acronym):
        """
        Args:
            value (str): A string representing the value which needs
            to be replaced by its short form.
            context (str): A string representing the context in which
            the long form needs to be replaced by the value.
            acronym (str): A string representing the short form
        Returns:
            str: The updated context after replacing full form with its acronym wrapped
            inside special tokens <start> and <end>. If the short form of the value is
              not found in the context, it returns 1.
        """
        result = " " + value + " "
        acronym = (
            self.config["start_tag"]
            + " "
            + acronym
            + " "
            + self.config["end_tag"]
            + " "
        )
        if result in context:
            return context.replace(result, acronym)
        return "1"

    def proper_full_form(self, context, value):
        """
        Returns a modified version of the `context` string with the
        `value` string replaced with its proper full form.
        Args:
        - context (str): a string representing the context in which `value` appears
        - value (str): a string representing full form for a given short form
        """
        value1 = " " + value + " "
        context = context.replace(f"{value} ", value1).replace(f"{value},", value1)
        context = context.replace(f"{value}.", value1)
        if value1 in context:
            return context
        if value in context and len(value) > 5:
            return context.replace(f"{value}", value1)
        return "100"

    def len_context(self, context, acronym):
        """
        Returns a modified version of the `context` string with the specified acronym
        in its wrapped from is surrounded by a certain number of characters before and after it.
        Args:
        - self: the object instance
        - context (str): a string representing the context in which the `acronym` appears
        - acronym (str): a string representing the acronym
        """
        backup = context
        acronym = (
            self.config["start_tag"]
            + " "
            + acronym
            + " "
            + self.config["end_tag"]
            + " "
        )
        len_cont = self.config["max_len"]
        match = re.search(acronym, context)
        if match:
            start1 = match.start()
            start = max(0, (start1 - len_cont))
            end = min(len(context) - 1, start1 + len(acronym) + len_cont)

            if not context[start].isspace():
                word_end = re.search(r"\s", context[start:])
                if word_end:
                    start += word_end.start()
            if not context[end].isspace():
                word_end = re.search(r"\s", context[end:])
                if word_end:
                    end += word_end.start()
            result = context[start:end]
            return context.replace(backup, result)
        return "1"

    def preprocess_text(self, text):
        """
        Preprocesses input text by converting it to lowercase and removing all punctuation marks.

        Args:
            - text (str): the input text to be preprocessed
        """
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return text

    def remove_similar_rows(self, data_frame, col, lst):
        """
        Removes similar rows from the input DataFrame using the cosine similarity
        between TF-IDF vectors for each group of rows with the same acronym.

        Args:
        - data_frame (pandas.DataFrame): the input DataFrame to
        be processed, with columns named 'acronym', 'value', and 'text'
        """
        # Preprocess the text column
        data_frame[col] = data_frame[col].apply(self.preprocess_text)
        # Create a TfidfVectorizer to generate TF-IDF vectors for each sentence
        tfidf_matrix = self.vectorizer.fit_transform(data_frame[col])

        # Compute the pairwise cosine similarity between the TF-IDF
        # vectors for each group of rows with the same acronym
        groups = data_frame.groupby(lst)
        if col == "text":
            keep_mask = np.ones((len(data_frame),), dtype=bool)
        else:
            # backup=data_frame[["acronym","value","full_form"]]
            keep_mask = np.zeros((len(data_frame),), dtype=float)
        for _, group in groups:
            indices = group.index
            group_cosine_similarities = cosine_similarity(tfidf_matrix[indices])
            # Mark all similar rows to the current row as duplicates within each group
            for i, index in enumerate(indices):
                if keep_mask[index] and col == "text":
                    for j in range(i + 1, len(indices)):
                        if group_cosine_similarities[i, j] >= self.similarity_threshold:
                            keep_mask[indices[j]] = False
                if col == "full_form_backup":
                    for j in range(i + 1, len(indices)):
                        if (
                            group_cosine_similarities[i, j]
                            >= self.config["cosine_similar_limit"]
                        ):
                            keep_mask[indices[j]] = group_cosine_similarities[i, j]
                            data_frame.loc[indices[j], "value"] = data_frame.loc[
                                indices[i], "value"
                            ]
        # Remove the similar rows from the DataFrame
        if col == "text":
            data_frame = data_frame[keep_mask]
            return data_frame

        backup = data_frame.copy()
        backup = backup[["acronym", "value", "full_form"]]
        data_frame["keep_mask"] = keep_mask
        data_frame.loc[:, "value"] = data_frame["value"].str.strip()
        backup.loc[:, "value"] = backup["value"].str.strip()
        backup.loc[:, "full_form"] = backup["full_form"]
        long_rows = data_frame[
            data_frame["value"].str.len() > self.config["max_full_form_length"]
        ]
        data_frame = data_frame.drop(index=long_rows.index)
        return data_frame, backup

    def preprocessed_dataset(self, data_frame, df_dict):
        """
        This method returns finalized dataframe after preprocessing
        Arg(s):
        dataframe: A pandas dataframe
        df_dict: A dataframe of acronym and its full forms

        """
        result = pd.merge(data_frame, df_dict, on="acronym", how="inner")
        result["value"] = result["value" + str("_y")]
        result["label"] = np.where(
            result["value" + str("_x")] == result["value" + str("_y")], 1, 0
        )
        result["sentence"] = (
            result["value"].map(
                lambda x: self.config["cls_token"]
                + " "
                + x
                + " "
                + self.config["separator_token"]
                + " "
            )
            + result["sentence"]
            + " "
            + self.config["separator_token"]
            + " "
        )
        data_frame2 = pd.DataFrame(
            result,
            columns=["acronym", "value", "sentence", "label", "correct_full_form"],
        )
        return data_frame2

    def process_dataset(self, data_frame, diction):
        """
        This method is used for modifying the context paragraph
        in the following format: [CLS]+long_form+[SEP]+context_paragraph+[SEP]
        Returns:
            dataframe
        """
        df_dict = pd.DataFrame(list(diction.items()), columns=["acronym", "value"])
        df_dict = df_dict.explode("value")

        df_dict["value"] = (
            df_dict["value"]
            .str.replace("(", "", regex=True)
            .str.replace(")", "", regex=True)
        )
        df_dict["value"] = df_dict["value"].apply(
            lambda x: x.replace(re.search(r"\[.*?\]", x).group(), "")
            if re.search("\[.*?\]", x)
            else x
        )
        df_dict["value"] = df_dict.apply(
            lambda row: self.replace_fullform_context(row["value"]), axis=1
        )
        df_dict = df_dict.drop(df_dict[df_dict["value"] == "1"].index)

        # dropping those rows where length of acronym and full_forms are same
        df_dict = df_dict[
            df_dict.apply(lambda x: len(x["acronym"]) != len(x["value"]), axis=1)
        ]
        df_dict["full_form_length"] = df_dict["value"].str.len()
        df_dict = df_dict.sort_values(
            "full_form_length", ascending=True
        )  # sort based on full_form_length
        df_dict = df_dict.drop("full_form_length", axis=1)
        df_dict = df_dict.reset_index(drop=True)
        df_dict["full_form_backup"] = df_dict["value"].str.strip()
        df_dict["full_form"] = df_dict["value"].str.strip()
        df_dict, backup = self.remove_similar_rows(
            df_dict, col="full_form_backup", lst="acronym"
        )
        df_dict = df_dict[["acronym", "value"]].drop_duplicates()
        df_dict = df_dict.reset_index(drop=True)
        data_frame["full_form"] = data_frame["value"]
        data_frame = data_frame.merge(backup, on=["acronym", "full_form"], how="inner")
        data_frame.loc[
            data_frame["full_form"] != data_frame["value_y"], "full_form"
        ] = data_frame["value_y"]
        data_frame = data_frame.drop(columns=["value_x", "value_y"]).rename(
            columns={"full_form": "value"}
        )
        data_frame["sentence"] = data_frame["context"]
        data_frame["correct_full_form"] = data_frame["value"].str.strip()
        result_train, result_test = train_test_split(
            data_frame,
            random_state=self.config["random_state"],
            test_size=self.config["test_size"],
        )
        result_valid, result_test = train_test_split(
            result_test,
            random_state=self.config["random_state"],
            test_size=self.config["test_size"],
        )
        train_data = self.preprocessed_dataset(result_train, df_dict)
        test_data = self.preprocessed_dataset(result_test, df_dict)
        valid_data = self.preprocessed_dataset(result_valid, df_dict)
        return train_data, test_data, valid_data

    def pipeline(self):
        """
        Returns a list of functions representing the data processing pipeline. The
        pipeline performs a sequence of data transformations on a DataFrame, using
        the instance methods of the current object.

        Returns:
        --------
        pipeline : list of callable
        """
        pipeline = [
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.acro_in_context_with_s(
                        row["acronym"], row["context"]
                    ),
                    axis=1,
                )
            ),
            lambda df: df.assign(
                value=df.apply(
                    lambda row: self.replace_fullform_context(row["value"]), axis=1
                )
            ),
            lambda df: df.drop(df[df["value"] == "1"].index),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.replace_fullform_context(row["context"]), axis=1
                )
            ),
            lambda df: df.drop(df[df["context"] == "1"].index),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.full_form_with_s(row["value"], row["context"]),
                    axis=1,
                )
            ),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.full_form_in_short(row["value"], row["context"]),
                    axis=1,
                )
            ),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.full_form_with_article(
                        row["value"], row["context"]
                    ),
                    axis=1,
                )
            ),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.proper_full_form(row["context"], row["value"]),
                    axis=1,
                )
            ),
            lambda df: df.drop(df[df["context"] == "100"].index),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.replace_full_form_with_acronym(
                        row["value"], row["context"], row["acronym"]
                    ),
                    axis=1,
                )
            ),
            lambda df: df.drop(df[df["context"] == "1"].index),
            lambda df: df.assign(
                context=df.apply(
                    lambda row: self.len_context(row["context"], row["acronym"]), axis=1
                )
            ),
        ]
        return pipeline

    def train_format(self):
        """this function creates the finalized preprocessed dataframe
        Returns: data_frame
        """
        data_f = self.create_dataframe_from_data()
        data_f.loc[:, "value"] = data_f["value"].str.strip()
        data_f["value"] = (
            data_f["value"]
            .str.replace("(", "", regex=True)
            .str.replace(")", "", regex=True)
        )
        data_f["value"] = data_f["value"].apply(
            lambda x: x.replace(re.search(r"\[.*?\]", x).group(), "")
            if re.search(r"\[.*?\]", x)
            else x
        )
        data_f = data_f.drop(data_f.loc[data_f["acronym"] == data_f["value"]].index)

        # remove repeated internal whitespace characters in column
        data_f["context"] = data_f["context"].str.replace(r"\s+", " ", regex=True)
        # Define the pipeline
        pipeline = self.pipeline()
        for func in pipeline:
            data_f = func(data_f)
        data_f = data_f.drop(data_f[data_f["context"] == "1"].index)
        data_f["text"] = data_f["context"]
        data_f = data_f.reset_index(drop=True)
        data_f = self.remove_similar_rows(data_f, col="text", lst=["acronym", "value"])
        train_data, test_data, valid_data = self.process_dataset(data_f, self.diction)
        return train_data, test_data, valid_data
