import argparse
import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel

class importedBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(importedBERT, self).__init__(config)
        self.bert = BertModel(config=config)

class myBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(myBERT, self).__init__(config)
        self.importedConfig = BertConfig.from_pretrained(args.model_name_or_path)
        self.importedBERT = importedBERT.from_pretrained(args.model_name_or_path, config=self.importedConfig)

        self.bert = BertModel(config=config)

        if args.num_hidden_layers > 12:
            print("******From the 1st layer to the 12th layer of the custom BERT would be replaced with to the pre-trained BERT layer while from the 13th layer would be randomly initialized.")
            layers_to_replace = [x for x in range(12)]
        else:
            layers_to_replace = [x for x in range(config.num_hidden_layers)]

        for layer in layers_to_replace:
            self.bert.base_model.encoder.layer[layer] = self.importedBERT.base_model.encoder.layer[layer]

        del self.importedBERT
        del self.importedConfig

    def forward(self, _input):
        sequence_output, pooled_output = self.bert(**_input)
        return sequence_output, pooled_output
