# encoding: utf-8

from itertools import izip
import logging
import numpy as np

random_seed = 42
logger = logging.getLogger(__name__)


class H5EmbeddingManager(object):
    def __init__(self, h5_path):
        import h5py
        f = h5py.File(h5_path, 'r')
        self.W = np.array(f['embedding'])
        logger.info("embedding data type=%s, shape=%s" % (type(self.W), self.W.shape))
        self.id2word = f['words_flatten'][0].split('\n')
        self.word2id = dict(izip(self.id2word, range(len(self.id2word))))

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def word_embedding_initialize(self, words_list, dim_size=300, scale=0.1, mode='pretrained', oov_init='random'):
        word2id = self.word2id
        W = self.W
        shape = (len(words_list), dim_size)
        np.random.seed(random_seed)
        if 'zero' == oov_init:
            W2V = np.zeros(shape, dtype='float32')
        elif 'one' == oov_init:
            W2V = np.ones(shape, dtype='float32')
        else:
            W2V = np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V[0, :] = 0
        if mode == 'random':
            return W2V
        in_vocab = np.ones(shape[0], dtype=np.bool)
        word_ids = []
        for i, word in enumerate(words_list):
            if word in word2id:
                word_ids.append(word2id[word])
            else:
                in_vocab[i] = False
        W2V[in_vocab] = W[np.array(word_ids, dtype='int32')][:, :dim_size]
        return W2V
