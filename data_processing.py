# begin script -------------------------------------------------------

""" data_processing.py
module containing classes and methods for processing data for
fine-tuning pre-trained bert.
"""

###     mark: edit __init_ of test data obj, we might not need it
###     and just inherit __init__ from the train data obj class.
###     reduce redudancy.

__author__ = "Christopher Garcia Cordova"

# imports ------------------------------------------------------------

import  numpy as np

from    pandas           import read_csv

import  pickle

from    sklearn.preprocessing\
                         import MultiLabelBinarizer

import  torch
from    torch.utils.data import Dataset, DataLoader

from    transformers     import BertTokenizerFast


# class def ----------------------------------------------------------

class Sequencer:

    """
    class for create mappings between unique strs and unique ints,
    and encoding and decoding from one to another.
    """

    def __init__(self, text, unk='[UNK]'):

        """
        class constructor.
            params:
                text: type: iterable(str):
                    a list of strings where each potentially
                    comprises a set of tokens.
            return: type: None
        """

        # record unk
        self.unk = unk

        # get unique tokens of text.
        self.vocab = self._get_vocabulary(text)

        # create str to int bijective mapping.
        self.tok2id =\
            {tok:id_ for id_, tok in enumerate(self.vocab)}

        # create inverse map.
        self.id2tok =\
            {id_:tok for tok, id_ in self.tok2id.items()}

    def __str__(self):

        """
        return the string representation of the current obj.
            params: type: None.
            return: type: None.
        """

        return str(self.tok2id)

    def __len__(self):

        """
        returns the number of unique tokens in the vocabulary
        of the current sequencer.
            params: type: None
            return:   type: None
        """

        return len(self.tok2id)

    def _get_vocabulary(self, text):

        """
        private helper function. tokens from a list(str) obj.
            params:
                text: type: list(str):
                    the text to find the unique tokens of.
            return: type: set:
                    a set of unique tokens.
        """
        
        unique_toks = set()

        # iterate of the list of strings and split them to
        # extract the unique tokens, by white space.
        for string in text:
            unique_toks.update(set(string.split()))

        return unique_toks
    
    def get_tok_idx(self, tok):

        """
        returns the idx of a given token if in the vocab of the
        sequencer.
            params:
                tok: type: str:
                    tok of interest
            return: type: int:
                idx of tok if present, -1 otherwise.
        """

        if tok in self.tok2id: return self.tok2id[tok]

        return -1

    def add_token(self, tok):

        """
        add tokens to the current sequencer.
            params:
                tok: type: str:
                    the token to add to the sequencer.
            return: type: None
        """

        # ensure tokens not already in sequencer vocab.
        if tok in self.tok2id: return

        # otherwise, add it.
        idx = len(self.tok2id)
        self.tok2id[tok] = idx
        self.id2tok[idx] = tok

    def encode(self, seq, unk_id=-1):

        """
        take a giving sec of tokens and mapping them to their
        unique ints.
            params:
                seq: type: list(str):
                    a list of tokens to map to ints.
                unk_id: type: int:
                    the default idx for OOV.
            return: type: list(int):
                    the respective encodings of the tokens into
                    ints.
        """

        int_rep = list()

        # iterate over and map each tok to its int.
        for tok in seq: 
            # ensure token is actually in the sequencer vocab.
            # if not, map it to a defualt idx.
            if tok not in self.tok2id:
                int_rep.append(unk_id)
                continue

            # otherwise, it is, map it.
            int_rep.append(self.tok2id[tok])

        return int_rep

    def decode(self, seq):

        """
        inverse of encode(), maps ints to their tokens.
            params:
                seq: type: list(int):
                    list of ints to map from.
            return: type: list(str):
                    list of strs mapped to.
        """

        str_rep = list()

        # iterate over each int and map it to its tok.
        for idx in seq:
            # ensure int is a possible mapping for a tok.
            # if not, map to a defualt unk tok.
            if idx not in self.id2tok:
                str_rep.append(self.unk)
                continue

            # otherwise, it is defined, map it to tok.
            str_rep.append(self.id2tok[idx])

        return str_rep

# class def ----------------------------------------------------------

class IntentTrainDataset(Dataset):

    """
    class for processing movie queiries for intent multi-label
    classification and tagging, specifically preparing this 
    data for fine-tuning a pre-trained language model.
    """

    def load(in_file):

        """
        class method for loading in a dataset obj. 
            params:
                in_file: type: str:
                    the relative or absolute of the file to
                    read the object from, searlized.
            return: type: IntentTrainDataset.
        """

        with open(in_file, 'rb') as f: data = pickle.load(f)

        return data 

    def __init__(self, file_path_or_data, tokenizer=None,
                     empty_class='no_relation',
                         text_col='utterances',
                             tag_col='IOB Slot tags',
                                 class_col='Core Relations'):

        """
        class constructor.
            params:
                file_path: type: str or pd.df:
                    as a string:
                    specifies the relative or absolute path of the
                    file containing the data, in csv format.
                    as a DataFrame:
                    contains pre-processed data to build the obj
                    from.
                tokenizer: type: transoformers.tokenizer:
                    -- optional --
                    default val: None
                    a tokenizer used by pre-trained bert, this
                    option is given only to use a tokenizer
                    with a particular config. don't init the
                    tokenizer, this is taken care of internally
                    here. otherwise, don't provide one. 
                empty_class: type: str:
                    -- optional --
                    default val: 'no_relation'
                    specify the value to map to when no classification
                    is to be made on part of the model.
                text_col: type: str:
                    -- optional --
                    default val: 'utterances'
                    name of the column holding the input text in the
                    csv file.
                tag_cal: type: str:
                    -- optional --
                    default val: 'IOB Slot tags'
                    name of the col with the tags. this col is not
                    allowed to have nan values, errors will occur
                    otherwise.
                class_col: type: str:
                    -- optional --
                    default val: 'Core Relations'
                    name of the col with the multi-labels. this
                    col is allowed to have nan values, this
                    text can possibly map to no relation at all.
            return: type: None
        """

        # read data from path, if provided.
        if isinstance(file_path_or_data, str):
            data = read_csv(file_path_or_data)

        # process each text example as list(str).
        self.texts =\
            [text.split() for text in data[text_col].tolist()]

        # process the tags for each example.
        self.tag_sequencer =\
            Sequencer(data[tag_col].tolist())
        self.tags =\
            [tags.split() for tags in data[tag_col].tolist()]

        # process the classifications for each example.
        intents = data[class_col].fillna(empty_class).tolist()
        self.labels = [labels.split() for labels in intents]

        # record name of empty class.
        self.empty_class = empty_class

        # initialize multilabel encoder.
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.labels)

        # if a bert tokenizer with special config not provided,
        # initialize default.
        if not tokenizer:
            tokenizer =\
                BertTokenizerFast.from_pretrained(
                        'bert-base-uncased',
                        do_lower_case=True
                    )

        # intialize the tokenizer.
        self.tokenizer =\
            tokenizer(
                self.texts,
                return_offsets_mapping=True,
                is_split_into_words=True,
                padding=True,
                truncation=True
            )

    def __getitem__(self, idx):

        """
        return a single training instance.
            params:
                idx: type: int:
                    the n-th instance of the current dataset, where
                    n = idx.
            return: type: dict[str] -> torch.tensor:
        """

        # get input ids, attention_mask, token_type_ids from
        # tokenizer.
        inst =\
            {
              key: torch.tensor(val[idx])\
              for key, val in self.tokenizer.items()
            }

        # get the gold tags for instance at idx.
        tags   = self.tags[idx]

        # the offset is a locates subtokens (produced by word-
        # piece) relative to the original token they were
        # dervied from.
        offset = self.tokenizer.offset_mapping[idx]

        # align sub-token offset with output tags, as wordpiece
        # produces more sub-tokens then there are tags after
        # it performs bpe.
        inst['tags'] =\
            torch.tensor(self.fix_tags_text_mismatch(tags, offset))

        # get multi, gold labels for instance at idx.
        inst['labels'] =\
            torch.tensor(
                self.encode_multilabel(self.labels[idx]),
                dtype=float
            )

        # pop off, don't pass this to model.
        inst.pop('offset_mapping')

        return inst

    def __len__(self):

        """
        return the size of the data -- the number of samples.
            params: type: None
            return: type: None
        """

        return len(self.texts)


    def exchange(self, other, also_tokenizer=False):

        """
        exchange the methods used by this obj for encoding and
        decoding IO with that of the other. this is useful
        when data is imbalanced and there are tags or labels
        observable in training but not in test data. used for
        preventing errors that might arise from this.
            params:
                other: type: IntentDataset:
                    the other dataset obj.
                also_tokenizer: type: bool:
                    default: False
                    specify whether to also exchange the
                    tokenizer method.
            return: type: None
        """

        # exachange encoding and decoding methods from
        # the other dataset obj for the the current obj
        # to use.
        if also_tokenizer:
            self.tokenizer = other.tokenizer
        self.tag_sequencer = other.tag_sequencer
        self.mlb           = other.mlb

    def num_labels(self):

        """
        returns the number of unique labels for multi-label class-
        ification.
            params: type: None
            return: type: None
        """

        return len(self.mlb.classes_)

    def num_tags(self):

        """
        returns the number of unique tags in the dataset.
            params: type: None
            return: type: None
        """

        return len(self.tag_sequencer)

    def fix_tags_text_mismatch(self, tags, offset):

        """
        bert tokenzizer is bpe based, and has an offset mapping
        field, this method fixes the potential mismatch between
        sub-tokens and tags (as tags are aligned with tokens
        before bpe). the offset mapping is an interable of
        2-tuples indicating the start position and end position of the
        sub-tokens relative to the original token derived from.
        possible positions are thus [0, len(original-token)-1]. this
        method fixes the mismatch by only encoding the sub-token that
        starts the original, ignoring the rest by setting there values
        to -100. 2-tuples (0, 0) also indicate special
        tokens used by bert such as cls, these are also set to -100.
            params:
                tags: type: list(str):
                    the raw tags to encode for a single example.
                offset: type: list(tuple(int, int)):
                    the offset mapping of the sub-tokens for the
                    single text example.
            return: type: list(int):
                    the fixed encoding rep of the raw tags.
        """

        # encode the tags into their indices.
        tags = self.tag_sequencer.encode(tags)

        # create an array of -100, the mask idx.
        encoded_tags = np.ones(len(offset), dtype=int) * -100

        # represent the offset mapping as an array, an np matrix.
        matrix_offset = np.array(offset)

        # set labels whose fist offset position is 0 and the second
        # is not 0.
        encoded_tags[
            (matrix_offset[:, 0] == 0) & (matrix_offset[:, 1] != 0)
        ] = tags

        return encoded_tags.tolist()

    def encode_multilabel(self, labels):

        """
        return the one-hot encoding of a set of labels for multi-
        label classification.
            params:
                labels: type: list(str)
                    a list of string, one for each labels that
                    a text sample maps to.
            return: type: np.array:
                    the labels as a one hot vector, where each
                    element is one if the class label is present,
                    0 otherwise.
        """

        # mlb expects an iterable of iterables, hence why labels is
        # but inside another list. it also returns a same, hence we
        # put off the return.
        return self.mlb.transform([labels]).tolist()[0]

    def decode_multilabel(self, labels):

        """
        return the labels encoded by a one-hot, inverse of encode
        multilabel.
            params:
                labels: type: list(list(str))
                    a list of list of labels text samples map to.
                    each element in the inner lists is a single
                    string corresponding to a single label.
            return: type: list(tuple(str))
                    the labels mapped from one hot encodings to
                    their original representation.
        """

        return self.mlb.inverse_transform(labels)

    def save(self, out_file='./pickled_dataobj'):

        """
        method for serializing the current dataset obj. see the
        static class method load() for loading it back.
            params:
                data: type: IntentTrainDataset:
                    and object instance to serialize.
                out_file: type: str:
                    optional
                    default val: ./pickled_dataobj
                    the relative or absolute path of the
                    the file to save the data to.
            return: type: None.
        """

        with open(out_file, 'wb') as f: pickle.dump(self, f)

# class def ----------------------------------------------------------

class IntentTestDataset(IntentTrainDataset):

    """
    class for processing testing data, inherits from IntentTrain-
    Dataset, we only make slight difference to the offset
    funcationality to recover tagger predictions.
    """

    def __init__(self, file_path_or_data, tokenizer,
                     empty_class='no_relation',
                         text_col='utterances'):

        """
        class constructor.
            params:
                file_path: type: str or pd.df:
                    as a string:
                    specifies the relative or absolute path of the
                    file containing the data, in csv format.
                    as a DataFrame:
                    contains pre-processed data to build the obj
                    from.
                tokenizer: type: transoformers.tokenizer:
                    a tokenizer used by a pre-trained language model
                    to tokenize the text with.
                empty_class: type: str:
                    default val: no_relation
                    specify the name of a dummy class, to stand in
                    for when no classifications are made when using
                    this data at inference.
                text_col: type: str:
                    default val: 'utterances'
                    name of the column holding the text in the
                    csv file.
            return: type: None

        """

        if isinstance(file_path_or_data, str):
            data = read_csv(file_path_or_data)

        # process each text example as list(str).
        self.texts =\
            [text.split() for text in data[text_col].tolist()]

        # keep track of what the original text was.
        self.txt_sequencer =\
            Sequencer(data[text_col].tolist())

        # record empty classification rep.
        self.empty_class = empty_class

        # init tokenizer.
        self.tokenizer =\
            tokenizer(
                self.texts,
                return_offsets_mapping=True,
                is_split_into_words=True,
                padding=True,
                truncation=True
            )

        # default values, fields of parent class.
        self.labels    = None
        self.tags      = None
        self.mlb       = None
        self.tag_sequencer = None

    def __getitem__(self, idx):

        """
        return a single training instance.
            params:
                idx: type: int:
                    the n-th instance of the current dataset, where
                    n = idx.
            return: type: dict[str] -> torch.tensor:
        """

        # get input ids, attention_mask, token_type_ids from
        # tokenizer.
        inst =\
            {
              key: torch.tensor(val[idx])\
              for key, val in self.tokenizer.items()
            }

        # record what the original token sequence was, before
        # BPE, useful for extractign exact predictions.
        text = self.texts[idx]
        offset = self.tokenizer.offset_mapping[idx]
        inst['text'] =\
            torch.tensor(self.fix_sub_text_mismatch(text, offset))

        # pop off, don't pass this to model.
        inst.pop('offset_mapping')

        return inst

    def fix_sub_text_mismatch(self, text, offset):

        """
        assuming the tokenzizer is bpe based, and has a offset mapping
        fields, this method fixes the potential mismatch between
        sub-tokens and output. the offset mapping is an interable of
        2-tuples indicating the start position and end position of the
        sub-tokens relative to the original word is was derived from.
        possible positions are thus [0, len(original-token)-1]. this
        method fixes the mismatch by only encoding the sub-token that
        starts the original, ignoring the rest by setting there values
        to -100. special tokens use 2-tuple is (0, 0) indicate special
        tokens and are also set to -100.
            params:
                text: typ: list(str):
                    the raw text.
                offset: type: list(tuple(int, int)):
                    the offset mapping of the sub-tokens for the
                    single text example.
            return: type: list(int):
                    a fixed lists that marks unimportant tokens.
        """

        # encode the tags into their indices.
        text = self.txt_sequencer.encode(text)

        # create an array of -100
        encoded_text = np.ones(len(offset), dtype=int) * -100

        # represent the offset mapping as an array, an np matrix.
        matrix_offset = np.array(offset)

        # set labels whose fist offset position is 0 and the second
        # is not 0.
        encoded_text[
            (matrix_offset[:, 0] == 0) & (matrix_offset[:, 1] != 0)
        ] = text 

        return encoded_text.tolist()

# end script ---------------------------------------------------------
