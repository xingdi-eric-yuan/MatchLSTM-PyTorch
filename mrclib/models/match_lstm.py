# encoding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import logging
import torch
import numpy as np

from ..layers.eric_temp_layers import BiLSTM, Embedding,\
    TimeDistributedRNN, TimeDistributedEmbedding, BiMatchLSTM, BoundaryDecoder
logger = logging.getLogger(__name__)


class MatchLSTMModel(torch.nn.Module):
    model_name = 'match_lstm'

    def __init__(self, model_config, data_specs):
        super(MatchLSTMModel, self).__init__()
        self.data_specs = data_specs
        self.model_config = model_config

        logger.debug("MODEL CONFIG: \n%s" % self.model_config)
        logger.debug("DATA CONFIG: \n%s" % [(k, v)
                                            for k, v in self.data_specs.iteritems()
                                            if type(v) in [int, basestring, float, np.int64, np.int32, np.float32,
                                                           tuple]])
        self.vocab_size = data_specs['n_voc']
        self.vocab_size_char = data_specs['n_voc_char']
        self.id2word = data_specs['id2word']
        self.read_config()
        self._def_layers()
        self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        logger.info("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        logger.info("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # embedding config
        config = self.model_config['embedding']['word_level']
        self.embed_path = config['path']
        self.embedding_size = config['embedding_size']
        self.embedding_type = config['embedding_type']
        self.embedding_dropout = config['embedding_dropout']
        self.embedding_word_dropout = config['embedding_word_dropout']
        self.embedding_trainable = config['embedding_trainable']
        self.embedding_oov_init = config['embedding_oov_init']

        config = self.model_config['embedding']['char_level']
        self.char_embedding_size = config['embedding_size']
        self.char_embedding_rnn_size = config['embedding_rnn_size']
        self.char_embedding_dropout = config['embedding_dropout']
        self.char_embedding_word_dropout = config['embedding_word_dropout']
        self.char_embedding_trainable = config['embedding_trainable']

        # model config
        config = self.model_config[self.model_name]
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.match_lstm_hidden_size = config['match_lstm_hidden_size']
        self.dropout_between_rnn_hiddens = config['dropout_between_rnn_hiddens']
        self.dropout_between_rnn_layers = config['dropout_between_rnn_layers']
        self.dropout_in_rnn_weights = config['dropout_in_rnn_weights']
        self.use_layernorm = config['use_layernorm']
        self.decoder_hidden_size = config['decoder_hidden_size']

        self.enable_cuda = self.model_config['scheduling']['enable_cuda']

    def _def_layers(self):

        # word embeddings
        self.word_embedding = Embedding(embedding_size=self.embedding_size,
                                        vocab_size=self.vocab_size,
                                        trainable=self.embedding_trainable,
                                        id2word=self.id2word,
                                        word_dropout_rate=self.embedding_word_dropout,
                                        dropout_rate=self.embedding_dropout,
                                        embedding_oov_init=self.embedding_oov_init,
                                        embedding_type=self.embedding_type,
                                        pretrained_embedding_path=self.embed_path,
                                        enable_cuda=self.enable_cuda)

        # char embeddings
        self.char_embedding = TimeDistributedEmbedding(Embedding(embedding_size=self.char_embedding_size,
                                                                 vocab_size=self.vocab_size_char,
                                                                 trainable=self.char_embedding_trainable,
                                                                 word_dropout_rate=self.char_embedding_word_dropout,
                                                                 dropout_rate=self.char_embedding_dropout,
                                                                 embedding_type='random',
                                                                 enable_cuda=self.enable_cuda))

        self.char_encoder = TimeDistributedRNN(rnn=BiLSTM(nemb=self.char_embedding_size, nhids=self.char_embedding_rnn_size,
                                                          dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                                                          dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                                                          dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                                                          use_layernorm=self.use_layernorm,
                                                          enable_cuda=self.enable_cuda))

        # lstm encoder
        self.encoder = BiLSTM(nemb=self.embedding_size + self.char_embedding_rnn_size[-1], nhids=self.rnn_hidden_size,
                              dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                              dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                              dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                              use_layernorm=self.use_layernorm,
                              enable_cuda=self.enable_cuda)

        self.match_lstm = BiMatchLSTM(input_p_dim=self.rnn_hidden_size[-1], input_q_dim=self.rnn_hidden_size[-1], nhids=self.match_lstm_hidden_size,
                                      dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                                      dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                                      dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                                      use_layernorm=self.use_layernorm,
                                      enable_cuda=self.enable_cuda)

        self.decoder_init_state_generator = torch.nn.Linear(self.match_lstm_hidden_size[-1], self.decoder_hidden_size)

        self.boundary_decoder = BoundaryDecoder(input_dim=self.match_lstm_hidden_size[-1], hidden_dim=self.decoder_hidden_size, enable_cuda=self.enable_cuda)

    def forward(self, input_story, input_question, input_story_char, input_question_char):
        # word embedding
        story_word_embedding, story_mask = self.word_embedding.forward(input_story)  # batch x time x emb
        question_word_embedding, question_mask = self.word_embedding.forward(input_question)  # batch x time x emb
        # char embedding
        story_char_embedding, story_char_mask = self.char_embedding.forward(input_story_char)  # batch x time x max_char x emb
        _, story_char_embedding, _ = self.char_encoder.forward(story_char_embedding, story_char_mask)  # batch x time x char_emb
        question_char_embedding, question_char_mask = self.char_embedding.forward(input_question_char)  # batch x time x max_char x emb
        _, question_char_embedding, _ = self.char_encoder.forward(question_char_embedding, question_char_mask)  # batch x time x char_emb
        # concat them together
        story_embedding = torch.cat([story_word_embedding, story_char_embedding], -1)
        question_embedding = torch.cat([question_word_embedding, question_char_embedding], -1)
        # encodings
        story_encoding, _, _ = self.encoder.forward(story_embedding, story_mask)
        question_encoding, _, _ = self.encoder.forward(question_embedding, question_mask)
        # match lstm
        story_match_encoding, story_match_encoding_last_state, _ = self.match_lstm.forward(story_encoding, story_mask, question_encoding, question_mask)
        # generate decoder init state using story match encoding last state (batch x hid)
        init_state = self.decoder_init_state_generator.forward(story_match_encoding_last_state)
        init_state = torch.tanh(init_state)  # batch x hid
        # decode
        output = self.boundary_decoder.forward(story_match_encoding, story_mask, init_state)  # batch x time x 2

        return output
