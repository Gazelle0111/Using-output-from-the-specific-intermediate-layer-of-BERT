import argparse
from transformers import BertConfig, BertTokenizerFast
import customBERT as csBERT


def main(args):
    sample_text = """Officials are set to announce details of B.C.'s latest restart plan on Tuesday as daily case counts continue to trend downward and hours after the last round of "circuit breaker" restrictions expired."""
    
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    
    myconfig = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_hidden_layers=args.num_hidden_layers
        )
    
    model = csBERT.myBERT.from_pretrained(args.model_name_or_path, config=myconfig, args=args)
    
    _bert_processed_input = tokenizer(sample_text, return_tensors='pt')

    sequence_output, pooled_output = model(_bert_processed_input)
    print("sequence_output")
    print(sequence_output)
    print("\npooled_output")
    print(pooled_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="Specific pretrained BERT model.")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="Number of hidden layers that will be used for custom BERT.")

    args = parser.parse_args()

    main(args)
