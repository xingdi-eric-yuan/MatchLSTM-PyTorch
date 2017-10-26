# A Pytorch Implementation of MatchLSTM for SQuAD (NOT FINISHED YET)
A simple pytorch implementation of MatchLSTM (Wang and Jiang, 2016) model for SQuAD (Rajpurkar et al., 2016) question answering.

## Tokenized SQuAD
A NLTK tokenized version of SQuAD dataset is included. Because the original test set is hidden, so I split a subset from training split as "valid" set, and use the original dev set as test. the splitting is on Wikipedia article level.

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

## LICENSE
[MIT][mit_license]

[pytorch_install]: http://pytorch.org/
[glove_download]: https://nlp.stanford.edu/projects/glove/
[squad_paper_link]: https://arxiv.org/abs/1606.05250
[match_lstm_paper_link]: https://arxiv.org/abs/1608.07905
[mit_license]: https://github.com/xingdi-eric-yuan/match_lstm_qa_pytorch/blob/debug/LICENSE