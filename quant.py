# begin script -------------------------------------------------------

""" inference.py
script for performing inference and a model and its quantized
version.
"""
# imports ------------------------------------------------------------

from models          import *
from data_processing import  *

# load data ----------------------------------------------------------

train = IntentTrainDataset.load('./temp/train_data.pkl')
test  = IntentTestDataset.load('./temp/test_data.pkl')

# load, compress, test model -----------------------------------------

model =\
    FineTunedBert(
        model_name_or_path='bert-base-uncased',
        num_hidden=4,
        dropout=0.2,
        num_classes=train.num_labels(), 
        num_tags=train.num_tags()
    )

model.load_state_dict(torch.load('./temp/finetuned-bert.pt'))
model = model.quantize()
model.test(test, batch_size=1)

# end script ---------------------------------------------------------
