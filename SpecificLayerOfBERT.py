import argparse
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, BertForSequenceClassification, AdamW, BertPreTrainedModel

class adoptedBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(adoptedBERT, self).__init__(config)
        self.bert = BertModel(config=config)

class myBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(myBERT, self).__init__(config)
        self.adoptedConfig = BertConfig.from_pretrained(args.model_name_or_path)
        self.adoptedBERT = adoptedBERT.from_pretrained(args.model_name_or_path, config=self.adoptedConfig)

        self.bert = BertModel(config=config)

        if args.num_hidden_layers > 12:
            layers_to_replace = [x for x in range(12)]
        else:
            layers_to_replace = [x for x in range(config.num_hidden_layers)]

        for layer in layers_to_replace:
            self.bert.base_model.encoder.layer[layer] = self.adoptedBERT.base_model.encoder.layer[layer]

        del self.adoptedBERT
        del self.adoptedConfig

    def forward(self, _input):
        sequence_output, pooled_output = self.bert(**_input)
        return sequence_output, pooled_output


def get_output_from_the_custom_BERT(args, tokenizer, _text):
    myconfig = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_hidden_layers=args.num_hidden_layers
        )

    model = myBERT.from_pretrained(args.model_name_or_path, config=myconfig)

    _bert_processed_input = tokenizer(_text, return_tensors='pt')

    sequence_output, pooled_output = model(_bert_processed_input)

    return sequence_output, pooled_output

def main(args):
    if args.num_hidden_layers > 12:
        print("******From the 1st layer to the 12th layer of the custom BERT would be replaced with to the pre-trained BERT layer while from the 13th layer would be randomly initialized.")

    sample_text = """Officials are set to announce details of B.C.'s latest restart plan on Tuesday as daily case counts continue to trend downward and hours after the last round of "circuit breaker" restrictions expired."""
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    sequence_output, pooled_output = get_output_from_the_custom_BERT(args, tokenizer, sample_text)

    print(sequence_output)
    print()
    print(pooled_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="Selecting specific pretrained BERT model.")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="Number of hidden layers that will be used for custom BERT")

    args = parser.parse_args()

    main(args)
