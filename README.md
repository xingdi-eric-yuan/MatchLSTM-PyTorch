# A Pytorch Implementation of MatchLSTM for SQuAD
A simple pytorch implementation of MatchLSTM model for SQuAD question answering.

## Tokenized SQuAD
A tokenized version of SQuAD dataset is included. Because the original test set is hidden, so I split a subset from training split as "valid" set, and use the original dev set as test. the splitting is on Wikipedia article level.

## Requirements
* Python 2.7
* Install Pytorch, follow [this][pytorch_install].
* Download pretrained GloVe embeddings from [here][glove_download].
* Run `pip install --requirement requirements.txt`

## First Time Running
* Run `python helpers/embedding_2_h5.py` to generate glove embedding h5 file.
* It will take some time first time running because it will generate an h5 file for SQuAD dataset

## Results
TODO

## References
* [SQuAD Paper][squad_paper_link]
* [MatchLSTM Paper][match_lstm_paper_link]

[pytorch_install]: http://pytorch.org/
[glove_download]: https://nlp.stanford.edu/projects/glove/
[squad_paper_link]: https://arxiv.org/abs/1606.05250
[match_lstm_paper_link]: https://arxiv.org/abs/1608.07905