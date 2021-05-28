# Overview
- This project aims to customize BERT to use the output of the specific intermediate layer of pre-trained BERT for certain target tasks. The number of hidden layers is the parameter that can be specified by the user and if the parameter is larger than 12, from the 1st to 12th layer is replaced to the pre-trained BERT while from 13th layer is randomly initialized.

# Brief description
- SpecificLayerOfBERT.py
> Output format
> - sequence_output: The output of the last hidden state where the shape is (batch_size, sequence_length, hidden_size)
> - pooled_output: The output of the last layer hidden state of the first token(CLS) processed by a linear layer and a Tanh activation function.

# Prerequisites
> - argparse
> - transformers

# Parameters
> - model_name_or_path(str, defaults to "bert-base-uncased"): Specific pretrained BERT model.
> - num_hidden_layers(int, defaults to 1): Number of hidden layers that will be used for custom BERT.

# References
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
