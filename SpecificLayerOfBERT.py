import argparse
import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel


class myBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(myBERT, self).__init__(config)
        self.bert = BertModel(config=config)

    def forward(self, _input):
        sequence_output, pooled_output = self.bert(**_input)
        return sequence_output, pooled_output
