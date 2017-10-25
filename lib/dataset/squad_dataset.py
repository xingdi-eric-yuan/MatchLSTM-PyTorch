import os
import codecs
import random
import logging
from tqdm import tqdm
from itertools import izip
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SquadDataset(object):
    dataset_name = 'squad'
    padding = '--PAD--'
    unknown = '--UNK--'
    eos = '--EOS--'
    bos = '--BOS--'

    def __init__(self, dataset_h5='squad_dataset.1.0.h5', data_path='squad_dataset.1.0/', ignore_case=False, **kwargs):
        super(SquadDataset, self).__init__(**kwargs)
        self.special_tokens = (self.padding, self.unknown, self.eos, self.bos)

        self.meta_data = dict()
        self.data_path = data_path
        self.ignore_case = ignore_case
        self.dataset_h5 = dataset_h5
        self._build()
        self.data = None
        self._init_h5()

    def _build(self):
        if not os.path.exists(self.dataset_h5):
            # build an hdf5 version of the dataset it the requested one doesn't exist
            self._caches = dict(
                _train_cache=dict(
                    story_word_id=[], question_word_id=[], answer_word_id=[], answer_heads=[], answer_tails=[]),
                _valid_cache=dict(
                    story_word_id=[], question_word_id=[], answer_word_id=[], answer_heads=[], answer_tails=[]),
                _test_cache=dict(
                    story_word_id=[], question_word_id=[], answer_word_id=[], answer_heads=[], answer_tails=[]),
            )
            self.meta_data['word2id'] = dict(zip(self.special_tokens, xrange(len(self.special_tokens))))
            self.meta_data['id2word'] = list(self.special_tokens)
            self.meta_data['word2id_char'] = dict(zip(self.special_tokens, xrange(len(self.special_tokens))))
            self.meta_data['id2word_char'] = list(self.special_tokens)
            self._scan_vocab()

    def _init_h5(self):
        '''
        Transfer data from h5 to Dataset object
        '''
        import h5py
        meta_data = {}
        f = h5py.File(self.dataset_h5, "r")
        dataset = f[self.dataset_name]
        for key in dataset.attrs:
            meta_data[key] = dataset.attrs[key]

        words_flatten = f['words_flatten'][0]
        id2word = words_flatten.split('\n')
        word2id = dict(izip(id2word, range(len(id2word))))
        meta_data['id2word'] = id2word
        meta_data['word2id'] = word2id
        del words_flatten
        meta_data['n_voc'] = len(meta_data['id2word'])

        words_flatten = f['words_flatten_char'][0]
        id2word = words_flatten.split('\n')
        word2id = dict(izip(id2word, range(len(id2word))))
        meta_data['id2word_char'] = id2word
        meta_data['word2id_char'] = word2id
        del words_flatten
        meta_data['n_voc_char'] = len(meta_data['id2word_char'])

        data = {}
        for key in dataset['data']:
            data[key] = dataset['data'][key]
        self.meta_data = meta_data
        self.data = data
        logger.info('finish init dataset with %s' % self.dataset_h5)
        return

    def _scan_vocab(self):
        self._handle_story()
        self._export_data_h5()
        del self._caches
        return

    def _export_data_h5(self):
        '''
        Transfer data from cache to h5 file
        '''
        import h5py
        f = h5py.File(self.dataset_h5, "w")
        grp = f.create_group(self.dataset_name)
        compress_option = dict(
            compression="gzip", compression_opts=9, shuffle=False)
        for key, value in self.meta_data.iteritems():
            value_type = type(value)
            # EXPORT WORDS_FLATTEN (VOCABULARY)
            if key == 'id2word' and value_type in [list]:
                words_flatten = '\n'.join(value)
                vocab_len = len(words_flatten)
                f.attrs['vocab_len'] = vocab_len
                dt = h5py.special_dtype(vlen=str)
                _dset_vocab = f.create_dataset(
                    'words_flatten', (1, ), dtype=dt, **compress_option)
                _dset_vocab[...] = [words_flatten]

            elif key.startswith('id2word_char') and value_type in [list]:
                words_flatten = '\n'.join(value)
                vocab_len = len(words_flatten)
                f.attrs['vocab_char_len'] = vocab_len
                dt = h5py.special_dtype(vlen=str)
                _dset_vocab = f.create_dataset(
                    'words_flatten_char', (1, ), dtype=dt, **compress_option)
                _dset_vocab[...] = [words_flatten]
            # EXPORT CACHE
            elif value_type in [np.ndarray]:
                _dset = grp.create_dataset(
                    key, value.shape, dtype=value.dtype, **compress_option)
                _dset[...] = value
            # EXPORT DATA STATS
            elif value_type in [int, float, str]:
                grp.attrs[key] = value
            # SKIPPING EVERYTHING ELSE
            else:
                logger.info('skipping %s, %s' % (key, value_type))

        sub_grp = grp.create_group('data')
        sub_grp_train = sub_grp.create_group('train')
        data_set_train = self._create_batch(self._caches['_train_cache'])
        for key, value in data_set_train.iteritems():
            _dset = sub_grp_train.create_dataset(
                key, value.shape, dtype=value.dtype, **compress_option)
            _dset[...] = value
        del data_set_train
        sub_grp_valid = sub_grp.create_group('valid')
        data_set_valid = self._create_batch(self._caches['_valid_cache'])
        for key, value in data_set_valid.iteritems():
            _dset = sub_grp_valid.create_dataset(
                key, value.shape, dtype=value.dtype, **compress_option)
            _dset[...] = value
        del data_set_valid
        sub_grp_test = sub_grp.create_group('test')
        data_set_test = self._create_batch(self._caches['_test_cache'])
        for key, value in data_set_test.iteritems():
            _dset = sub_grp_test.create_dataset(
                key, value.shape, dtype=value.dtype, **compress_option)
            _dset[...] = value
        del data_set_test
        f.flush()
        f.close()

    def _handle_story_split(self, caches, file_path):
        _story_word_id = caches['story_word_id']
        _question_word_id = caches['question_word_id']
        _answer_heads = caches['answer_heads']
        _answer_tails = caches['answer_tails']
        _answer_word_id = caches['answer_word_id']

        story_file = file_path + "-v1.1-story.txt"
        question_file = file_path + "-v1.1-question.txt"
        answer_range_file = file_path + "-v1.1-answer-range.txt"
        with codecs.open(story_file, mode='r', encoding='utf-8', errors='ignore') as story_reader,\
                codecs.open(question_file, mode='r', encoding='utf-8', errors='ignore') as question_reader,\
                codecs.open(answer_range_file, mode='r', encoding='utf-8', errors='ignore') as answer_range_reader:
            for i, (_story, _question, _a_ranges) in enumerate(tqdm(zip(story_reader, question_reader, answer_range_reader))):
                _story, _question, _a_ranges = _story.strip(), _question.strip(), _a_ranges.strip()
                if len(_a_ranges) <= 0:
                    continue

                story_id = self._words_to_ids(_story.split())
                question_id = self._words_to_ids(_question.split())

                _a_ranges = _a_ranges.split(" ||| ")
                if len(_a_ranges) <= 0:
                    continue
                # answer word id: all answers available
                answer_words = []
                for arange in _a_ranges:
                    head, tail = arange.split(':', 1)
                    answer_words.append(story_id[int(head): int(tail)])

                # answer ranges: if multiple answers, use first one
                head, tail = _a_ranges[0].split(':', 1)
                head, tail = int(head), int(tail)
                answer_head = [1 if i == head else 0 for i in range(len(story_id))]
                answer_tail = [1 if i == tail - 1 else 0 for i in range(len(story_id))]

                self._get_char_vocab(_story)
                self._get_char_vocab(_question)

                _story_word_id.append(story_id)
                _question_word_id.append(question_id)
                _answer_heads.append(answer_head)
                _answer_tails.append(answer_tail)
                _answer_word_id.append(answer_words)

        combined = zip(caches['story_word_id'],
                       caches['question_word_id'],
                       caches['answer_heads'],
                       caches['answer_tails'],
                       caches['answer_word_id'])
        random.shuffle(combined)
        caches['story_word_id'],\
            caches['question_word_id'],\
            caches['answer_heads'],\
            caches['answer_tails'],\
            caches['answer_word_id'] = zip(*combined)

    def _words_to_ids(self, words):
        word2id = self.meta_data['word2id']
        id2word = self.meta_data['id2word']
        ids = []
        for word in words:
            if self.ignore_case:
                if word not in self.special_tokens:
                    word = word.lower()
            try:
                ids.append(word2id[word])
            except KeyError:
                w_id = len(id2word)
                word2id[word] = w_id
                id2word.append(word)
                ids.append(w_id)
        return ids

    def _get_char_vocab(self, _input_string, preprocess=lambda x: x):
        word2id = self.meta_data['word2id_char']
        id2word = self.meta_data['id2word_char']
        _input_string = preprocess(_input_string)

        for ch in list(_input_string):
            if ch not in word2id:
                w_id = len(id2word)
                word2id[ch] = w_id
                id2word.append(ch)

    def _handle_story(self):

        cache_train, cache_valid, cache_test = \
            self._caches['_train_cache'],\
            self._caches['_valid_cache'],\
            self._caches['_test_cache']
        self._handle_story_split(
            cache_train, self.data_path + 'train')
        self._handle_story_split(
            cache_valid, self.data_path + 'valid')
        self._handle_story_split(
            cache_test, self.data_path + 'dev')

        # populate data stats in meta-data
        meta_data = self.meta_data

        def _max_len(field):
            return max(
                map(len,
                    cache_train[field] +
                    cache_valid[field] +
                    cache_test[field]))

        meta_data['max_nb_words_story'] = _max_len('story_word_id') + 1
        meta_data['max_nb_words_question'] = _max_len('question_word_id') + 1

        meta_data['max_nb_answer'] = _max_len('answer_word_id') + 1

        meta_data['max_nb_words_answer'] = max(
            map(len, [word for sent in cache_train['answer_word_id'] +
                      cache_valid['answer_word_id'] +
                      cache_test['answer_word_id'] for word in sent])) + 1

        meta_data['nb_train'] = len(cache_train['story_word_id'])
        meta_data['nb_valid'] = len(cache_valid['story_word_id'])
        meta_data['nb_test'] = len(cache_test['story_word_id'])
        meta_data['nb_pairs'] = meta_data['nb_train'] + meta_data['nb_valid'] + meta_data['nb_test']

        logger.info("nb_train=%s, nb_valid=%s, nb_test=%s" %
                    (meta_data['nb_train'], meta_data['nb_valid'], meta_data['nb_test']))
        logger.info("nb_pairs=%s, voc=%s, max_nb_words_story=%s, max_nb_words_question=%s, max_nb_answer=%s, max_nb_words_answer=%s" %
                    (meta_data['nb_pairs'],
                     len(meta_data['id2word']),
                     meta_data['max_nb_words_story'],
                     meta_data['max_nb_words_question'],
                     meta_data['max_nb_answer'],
                     meta_data['max_nb_words_answer']))

    def pad_sequences(self, sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
        '''Pads each sequence to the same length:
        the length of the longest sequence.
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen.
        Truncation happens off either the beginning (default) or
        the end of the sequence.
        Supports post-padding and pre-padding (default).
        # Arguments
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than
                maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        # Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x

    def _create_batch(self, caches):
        story, question = caches['story_word_id'], caches['question_word_id']
        head, tail, answer = caches['answer_heads'], caches['answer_tails'], caches['answer_word_id']

        head = self.pad_sequences(head, maxlen=self.meta_data['max_nb_words_story'], padding='post').astype('int32')
        tail = self.pad_sequences(tail, maxlen=self.meta_data['max_nb_words_story'], padding='post').astype('int32')
        head = np.reshape(head, head.shape + (1,))
        tail = np.reshape(tail, tail.shape + (1,))
        pad_answer = np.zeros((len(answer), self.meta_data['max_nb_answer'], self.meta_data['max_nb_words_answer']), dtype='int32')
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                nb_w = len(answer[i][j])
                pad_answer[i, j, :nb_w] = answer[i][j]

        ret_dict = {
            'input_story': self.pad_sequences(
                story,
                maxlen=self.meta_data['max_nb_words_story'], padding='post').astype('int32'),
            'input_question': self.pad_sequences(
                question,
                maxlen=self.meta_data['max_nb_words_question'], padding='post').astype('int32'),
            'answer_ranges': np.concatenate([head, tail], -1),
            'input_answer': pad_answer
        }
        return ret_dict

    def get_data(self, train_size=0, valid_size=0, test_size=0):
        nb_train = train_size if train_size > 0 else self.meta_data['nb_train']
        nb_valid = valid_size if valid_size > 0 else self.meta_data['nb_valid']
        nb_test = test_size if test_size > 0 else self.meta_data['nb_test']
        batches_train, batches_valid, batches_test = \
            self.data['train'], self.data['valid'], self.data['test']
        logger.info(
            "Get %s train, %s valid, %s test" % (nb_train, nb_valid, nb_test))
        keys_input = batches_train.keys()
        train = {key: batches_train[key][:nb_train] for key in keys_input}
        valid = {key: batches_valid[key][:nb_valid] for key in keys_input}
        test = {key: batches_test[key][:nb_test] for key in keys_input}
        return [train, valid, test]
