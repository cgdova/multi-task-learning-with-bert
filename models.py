# begin script -------------------------------------------------------

"""
module holding important class defs for building a transformer for
class and tagging joint prediction.
"""

__author__ = 'Christopher Garcia Cordova'

# imports ------------------------------------------------------------

from   evaluation       import *

import pandas as pd

import torch
from   torch            import nn
from   torch.nn         import BCEWithLogitsLoss, CrossEntropyLoss
from   torch.utils.data import Dataset, DataLoader
from   torch.nn         import Dropout
from   torch            import quantization

from   tqdm             import tqdm

from   transformers     import BertModel, BertTokenizer,\
                               BertTokenizerFast

# class def ----------------------------------------------------------

class FineTunedBert(nn.Module):

    """
    class defining a model based on pre-trained Bert, for joint
    prediction on classification and tagging.
    """

    def __init__(self, model_name_or_path, dropout, 
                     num_classes, num_tags, num_hidden=1):

        """
        class constructor. build a Bert-based model for joint
        prediction, classification and tagging.
            paramtrs:
                model_name_or_path: type: str:
                    provide the name of the particular Bert model
                    to use options can be found here:
                        https://huggingface.co/transformers/
                        pretrained_models.html
                    alternativelly, provide a path to a file holding
                    a pre-trained Bert model.
                dropout: type: float:
                    a number in [0, 1), indicating dropout rate.
                num_class: type: int:
                    the size of the output space for classfication.
                num_tags: type: int:
                    the size of the output space for tagging.
            return: none
        """

        # initialize model with Module.
        super(FineTunedBert, self).__init__()

        # load pre-trained model.
        self.bert_model =\
            BertModel.from_pretrained(model_name_or_path)

        # intialize dropout rate.
        self.dropout = Dropout(dropout)

        # record the number of hidden layers to concat and feed
        # to linear layers.
        self.num_hidden  = num_hidden

        # record the output spaces for the dual tasks.
        self.num_classes = num_classes
        self.num_tags    = num_tags

        # intialize a linear layer for classification.
        self.classifier  =\
            nn.Linear(
                self.bert_model.config.hidden_size * num_hidden,
                num_classes
            )

        # initialize another layer for tagging.
        self.tagger =\
            nn.Linear(self.bert_model.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask, token_type_ids,
                    gold_labels=None, gold_tags=None):

        """
        method defining the forward pass for Bert-based model
            paramtrs:
                input_ids: type: torch.tensor:
                    the sub-tokens mapped to their indices.
                attention_mask: type: torch.tensor:
                    bool tensor what tokens are attentable.
                token_type_ids: type: torch.tensor:
                    mask for segmenting to different sents passed
                    to the model.
                gold_labels: type: torch.tensor:
                    the gold labels for the given instance.
                    used at training.
                gold_tags: type: torch.tensor:
                    the gold tags for the given instance.
                    used at training.
            return: type: 
        """

        # get the encoding from bert.
        last_hidden_state, pooler_output, hiddens=\
            self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=False
            )

        # get the last n hidden representations of BERT
        # concatenate them along the colspace, and extract
        # the representation of the [CLS] token.
        if self.num_hidden > 1:
            pooler_output =\
                torch.cat(
                    [enc for enc in hiddens[-self.num_hidden:]],
                    dim=-1
                )[:,0,:]

        # feed encoding to multi-label classifier, for raw preds.
        labels_logits =\
            self.classifier(self.dropout(pooler_output))

        # feed encoding to tagger, for raw preds.
        tags_logits =\
            self.tagger(self.dropout(last_hidden_state))

        # check whether gold labels provided, if so, assume
        # we're in training.
        if gold_labels is not None:
            # initialize loss of multi-label classification
            loss_func = BCEWithLogitsLoss() 

            # compute loss on preidictions against gold.
            classification_loss =\
                loss_func(labels_logits, gold_labels)

        # else, if no gold labels provided, assume we're are not
        # in training, loss 0 during eval.
        else: classification_loss = torch.tensor(0)

        # check whether gold tags are provided, if so, assume
        # we're in training.
        if gold_tags is not None:
            # check whether we can only attend to particular
            # tokens when computing loss, ignore rest.
            if attention_mask is not None:
                # initialize loss.
                loss_func = CrossEntropyLoss(ignore_index=-100)

                # find elements to compute the loss on given the
                # attention mask. bool tensor.
                active_loss = (attention_mask.view(-1) == 1)

                # reduce the batch to a matrix, and pick out only 
                # those # logits that the attention masks says are
                # attendable. so matrix is of shape[num attenble
                # inputs by num of possible tags.
                active_logits =\
                    tags_logits.view(-1, self.num_tags)[active_loss]

                # gold tags is matrix of shape batch size by max len
                # input in batch. we reduce change the shape to be
                # a vector whose length is times the original dims
                # of the aformentioned matrix. then we only pick
                # out those tags that correspond to attenable tokens.
                active_labels =\
                    gold_tags.view(-1)[active_loss]

                # now we compare the extracted logits with the gold
                # logits: [num_attenanble_tok, num_possible_tags]
                # gold:   [num_attenable_tok]
                tagging_loss =\
                    loss_func(active_logits, active_labels)

            else:
                # if no attention mask provided, consider all
                # tokens for the loss.
                taggin_loss=\
                    loss_func(
                        tags_logits.view(
                            -1,
                            self.num_classes
                        ),
                        gold_tags.view(-1)
                    )

        # if no gold tags provided, assume not in training, loss is
        # zero during eval.
        else: tagging_loss = torch.tensor(0)

        return labels_logits, tags_logits, classification_loss,\
                   tagging_loss 

    def tune(self, train, optimizer, epochs=3, batch_size=4,
                 valid=None):

        """
        tune pre-trained language model for custom dataset, for
        joint multi-label classification and tagging. training
        is batched.
            paramtrs:
                train: type: data_processing.IntentDataset:
                    torch dataset obj with addition function-
                    nality.
                optimizer: type: transformers.optim
                    an optimizer from the transformers package.
                epochs: type: int:
                    default val: 3
                    the number of epochs to train for.
                batch: type: int:
                    default val: 4
                    size of batch, for batch training.
                valid: type: data_processing.IntentDataset:
                    default val: None
                    torch dataset obj with additional function-
                    nality.
            return: type: None
        """

        # initialize a dataloader.
        train =\
            DataLoader(train, batch_size=batch_size, shuffle=True)

        # enter training for given number of epochs.
        for epoch in range(epochs):
            # specify model is in training, we do it for each epoch
            # since validattion might be done.
            self.train()

            # extract batch for training.
            for batch in tqdm(train):
                # set the gradients to zero.
                optimizer.zero_grad()

                # get the token ids to feed to model as input.
                input_ids = batch['input_ids']

                # get which tokens are actually attenable, to
                # not be ignored.
                attention_mask = batch['attention_mask']

                # get token separation regarding segements, in
                # this case, marks attenable tokens.
                token_type_ids = batch['token_type_ids']

                # get the gold tags.
                tags = batch['tags']

                # get the gold labels.
                labels = batch['labels']

                # perform forward pass on model.
                outputs =\
                    self(
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        gold_labels=labels,
                        gold_tags=tags
                    )

                # get the loss on the tagger and classifier.
                tagger_loss, classifier_loss = outputs[2], outputs[3]

                # we need to retain graph when working with multiple
                # losses. perform backward pass.
                tagger_loss.backward(retain_graph=True)
                classifier_loss.backward()

                # update parameters, take a step in the direction
                # of the gradients with given optim.
                optimizer.step()
                
            # check whether validation set was provided, if so,
            # validate model.
            if valid: self.validate(valid, batch_size)

    def validate(self, valid, batch_size=4):

        """
        method for vaidating the performance of the model.
            paramtrs:
                valid: type: data_processing.IntentDataset:
                    torch dataset obj with addition function-
                    ality
                batch: type: int:
                    specifies the size of the batch during eval.
                metric: Type: Callable:
        """

        # specify model is being evaluated, no backprop.
        self.eval()

        # init data loader.
        loader = DataLoader(valid, batch_size=batch_size)

        # record all classifications and taggings.
        all_clf_pred = list(); all_clf_gold = list()
        all_tgr_pred = list(); all_tgr_gold = list()

        # for recording performance.
        clf_f1 = list(); tgr_f1 = list()

        # iterate over dev batch.
        print('validating:')
        for batch in tqdm(loader):
            # get the encodings of the input texts in batch.
            input_ids = batch['input_ids']

            # get which tokens are actually attenable, not to be
            # ignored.
            attention_mask = batch['attention_mask']

            # get which tokens belong to particular segments. in
            # our case this aligns with which toks are attenable.
            token_type_ids = batch['token_type_ids']

            # get the gold tag labels.
            gold_tags = batch['tags']

            # get gold labels.
            gold_labels = batch['labels']

            # do a forward pass with the model.
            outputs =\
                self(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    gold_labels=gold_labels,
                    gold_tags=gold_tags
                )

            # get the raw logits scores for both tasks.
            classifier_logits, tagger_logits = outputs[0], outputs[1]

            # perform a sigmoid activation over the raw logits.
            classifier_sigmoid =\
                torch.sigmoid(classifier_logits)

            # sigmoid maps from [0, 1], so we can simply use round to
            # pick out only those values above .5 from sigmoid act.
            pred_classifier =\
                torch.round(classifier_sigmoid)

            # retrieve predicted classes, non-numeric.
            pred_classes =\
                valid.decode_multilabel(
                    pred_classifier.detach().numpy()
                )

            # as above, so below.
            gold_classes =\
                valid.decode_multilabel(
                    gold_labels.detach().numpy()
                )

            # record all predictions of classifier and golds.
            all_clf_pred.extend(pred_classes)
            all_clf_gold.extend(gold_classes)

            # do the same now for the tagger.
            tags_softmax = torch.softmax(tagger_logits, dim=-1)
            pred_tagger  = torch.argmax(tags_softmax, dim=-1)

            # decode predicted tags ignoring those marked -100 in
            # gold tags.
            pred_classes =\
                list(
                    valid.tag_sequencer.decode(pred.tolist())\
                    for pred in\
                    pred_tagger[gold_tags != -100].unsqueeze(0)
                )

            # decode gold tags in a similar fashion.
            gold_classes =\
                list(
                    valid.tag_sequencer.decode(gold.tolist())\
                    for gold in\
                    gold_tags[gold_tags != -100].unsqueeze(0)
                )

            # record tagging predictions and golds
            all_tgr_pred.extend(pred_classes)
            all_tgr_gold.extend(gold_classes)

        # report performance to out stream.
        print('tagger report:')
        print(classification_report(all_tgr_gold, all_tgr_pred))
        print('classifier report:', end=' ')
        print(f1_score(all_clf_gold, all_clf_pred, intent=True))

    def test(self, test_data, batch_size=4, tag_col='IOB Slot tags',
                 clf_col='Core Relations', rm_empty=True,
                     out_path='./predictions.txt'):

        """
        method for testing the model on test/inference data.
        writes predictions to default or sepcified out file
        in tsv file.
            paramtrs:
                test_data: type: IntentDataset:
                    the dataset obj, see module data_processing.
                batch_size: type: int:
                    the the batch size at which to test the
                    model at each foward pass.
                tag_col: type: str:
                    default val: IOB Slot tags
                    the column to format the tagger predictions
                    into, in the resulting tsv.
                clf_col: type: str:
                    default val: Core Realtions
                    the column to format the classifier's
                    predictions into in the resulting tsv.
                rm_empty: type: bool:
                    default val: True
                    specify whether to remove empty classifica-
                    tions from tsv.
                out_path: type: str:
                    default val: predictions.txt
                    specifies the location to write a tsv file
                    of the model predictions.
            return: type: None
        """

        # specify model is being evaluated, no backprop.
        self.eval()

        # init data loader.
        loader = DataLoader(test_data, batch_size=batch_size)

        # record all classifications and taggings.
        all_clf_pred = list()
        all_tgr_pred = list()

        # iterate over dev batch.
        print('testing:')
        for batch in tqdm(loader):
            # get the encodings of the input texts in batch.
            input_ids = batch['input_ids']

            # get which tokens are actually attenable, not to be
            # ignored.
            attention_mask = batch['attention_mask']

            # get which tokens belong to particular segments. in
            # our case this aligns with which toks are attenable.
            token_type_ids = batch['token_type_ids']

            # get which tokens should be ignore, which extracted.
            text = batch['text']

            # do a forward pass with the model.
            outputs =\
                self(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

            # get the raw logits scores for both tasks.
            classifier_logits, tagger_logits = outputs[0], outputs[1]

            # perform a sigmoid activation over the raw logits.
            classifier_sigmoid =\
                torch.sigmoid(classifier_logits)

            # sigmoid maps from [0, 1], so we can simply use round to
            # pick out only those values above .5 from sigmoid act.
            pred_classifier =\
                torch.round(classifier_sigmoid)

            # retrieve predicted classes, non-numeric.
            pred_classes =\
                test_data.mlb.inverse_transform(
                    pred_classifier.detach().numpy()
                )

            # record all predictions of classifier.
            all_clf_pred.extend(
                ' '.join(list(pred)) for pred in pred_classes
            )

            # do the same now for the tagger.
            tags_softmax = torch.softmax(tagger_logits, dim=-1)
            pred_tagger  = torch.argmax(tags_softmax, dim=-1)

            # decode predicted tags ignoring those marked -100 in
            # gold tags, for each text.
            pred_classes =\
                [
                    test_data.tag_sequencer.decode(
                        pred[t != -100].tolist()
                    )\
                    for pred, t in zip(pred_tagger, text)
                ]

            # record tagging predictions
            all_tgr_pred.extend(
                ' '.join(list(pred)) for pred in pred_classes
            )

        # initialize pd.df.
        all_preds =\
            pd.DataFrame(
                {tag_col: all_tgr_pred, clf_col: all_clf_pred}
            )

        # check whether to remove empty predictions.
        if rm_empty:
            all_preds.replace(
                {clf_col: ''}, 
                value=test_data.empty_class,
                inplace=True
            )

        # write df to file. 
        all_preds.to_csv(
            out_path,
            sep='\t',
            header=False,
            index=False
        )

    def save(self, path='./finetuned-bert.pt'):

        """
        method for saving the current model instance.
            params:
                path: type: str:
                    the relative or absolute path at which to
                    save the model.
            return: type: None.
        """

        torch.save(self.state_dict(), path)

    def size(self, precision=32):

        """
        returns the approximate size of the model in terms
        of megabytes. only includes parameters/weights for which
        requires_grad is true.
            params:
                precision: type: int:
                    optional
                    default val, 32.
                    the number of bits for each model
                    parameter, usualy float 32.
            return: type: int:
                    the approximate number of bytes of
                    the model.
        """

        num_params =\
            sum(
                p.numel()
                for p in self.parameters() if  p.requires_grad
            )

        bits = num_params * precision
        mgb  = bits // 8e+6

        return mgb

    def quantize(self):

        """
        return a quantized version the current model instance.
        converting from float32 to int8.
            params: type: None.
            return: type: None.
        """

        q_model =\
            quantization.quantize_dynamic(
                self,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        return q_model

# end script ---------------------------------------------------------








