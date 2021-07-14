# being script -------------------------------------------------------

""" bert.py
example script of how to fine-tune pre-trained bert to perform
multi-task sequence classification and tagging for relation
extraction.
"""

__author__ = "Christopher Garcia Cordova"

# imports ------------------------------------------------------------

from models          import *
from data_processing import *
from transformers    import AdamW

# process data -------------------------------------------------------

# init wordpiece tokenizer.
tokenizer =\
    BertTokenizerFast.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

# read in data from file, and save.
train = IntentTrainDataset('./data/train.csv', tokenizer)
train.save('./temp/train_data.pkl')

# load in test data.
test= IntentTestDataset('./data/test.csv', tokenizer)
test.exchange(train)
test.save('./temp/test_data.pkl')

# init, train, save, test model --------------------------------------

model =\
    FineTunedBert(
        model_name_or_path='bert-base-uncased',
        num_hidden=4,
        dropout=0.2,
        num_classes=train.num_labels(), 
        num_tags=train.num_tags()
    )

# W in AdamW stands for 'weight decay fix'.
optim = AdamW(model.parameters(), lr=5e-5)

# fine-tune model on custom dataset.
model.tune(
    train=train,
    optimizer=optim,
    epochs=4,
)

# we save the model to specified path.
model.save('./temp/finetuned-bert.pt')

# test model on test data, run inference.
model.test(test, batch_size=1)

# end script ---------------------------------------------------------
