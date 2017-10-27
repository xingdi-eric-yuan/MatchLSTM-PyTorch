# coding=utf-8
import numpy as np
import torch
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue

from squad_eval import metric_max_over_ground_truths, exact_match_score, f1_score


def torch_model_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_model_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = torch.nn.modules.module._addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


############################################
# evaluation
############################################
def accuracy_eval(graph, _generator, batch_size, data_size):

    number_batch = data_size // batch_size
    if data_size % batch_size > 0:
        number_batch += 1

    exact_match = total = 0.0
    for i in range(number_batch):
        x, y = _generator.next()
        y_pred = graph.predict_on_batch(x)  # nb x n_class
        pred = np.argmax(y_pred, 1)  # nb
        gold = y[0][:, 0]  # nb
        same = np.equal(pred, gold).astype('int32')
        exact_match += np.sum(same)

    total = data_size
    exact_match = 100.0 * exact_match / total
    return exact_match


############################################
# generator
############################################
def shuffle_data_dict(data_dict):
    '''
    Shuffle each data array in the named array dict.
    -- Assume all input arrays have same array.shape[0].
    '''
    ary_len = data_dict.values()[0].shape[0]
    rand_perm = np.random.permutation(ary_len)
    for k, v in data_dict.items():
        # permute input_dict[k]
        data_dict[k] = v.take(rand_perm, axis=0)
    return data_dict


def trim_batch(batch, trim_margin=None):
    # for post padding
    # batch.shape: N * n_words
    if trim_margin is None:

        batch_temp = batch[:, ::-1]
        batch_temp = np.cumsum(batch_temp, axis=1)
        batch_temp = batch_temp[:, ::-1]
        zero = batch_temp == 0
        z_index = np.argmax(zero, axis=1)
        trim_margin = np.max(z_index)
    return batch[:, :trim_margin + 1], trim_margin


def generator_queue(generator, max_q_size=10, wait_time=0.05, nb_worker=1):
    '''Builds a threading queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`.
    '''
    q = queue.Queue()
    _stop = threading.Event()

    def data_generator_task():
        while not _stop.is_set():
            try:
                if q.qsize() < max_q_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    q.put(generator_output)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                raise

    generator_threads = [threading.Thread(target=data_generator_task)
                         for _ in range(nb_worker)]

    for thread in generator_threads:
        thread.daemon = True
        thread.start()

    return q, _stop


def random_generator(data_dict, input_keys, output_keys, batch_size, bucket_size=-1, sort_by=None, trim_function=None,
                     random_shuffle=True, char_level_func=None, word_id2word=None, char_word2id=None, enable_cuda=False):
    if bucket_size == -1:
        bucket_size = batch_size * 100
    sample_count = None
    for k, v in data_dict.items():
        if sample_count is None:
            sample_count = v.shape[0]
        if not (sample_count == v.shape[0]):
            raise Exception('Mismatched sample counts in data_dict.')

    if bucket_size > sample_count:
        bucket_size = sample_count
        print('bucket_size < sample_count')
    # epochs discard dangling samples that won't fill a bucket.
    buckets_per_epoch = sample_count // bucket_size
    if sample_count % bucket_size > 0:
        buckets_per_epoch += 1

    while True:
        # random shuffle
        if random_shuffle:
            data_dict = shuffle_data_dict(data_dict)

        for bucket_num in range(buckets_per_epoch):
            # grab the chunk of samples in the current bucket
            bucket_start = bucket_num * bucket_size
            bucket_end = bucket_start + bucket_size
            if bucket_start >= sample_count:
                continue
            bucket_end = min(bucket_end, sample_count)
            current_bucket_size = bucket_end - bucket_start
            bucket_idx = np.arange(bucket_start, bucket_end)
            bucket_dict = {k: v.take(bucket_idx, axis=0) for k, v in data_dict.iteritems()}

            if sort_by is not None:
                non_zero = bucket_dict[sort_by]
                while non_zero.ndim > 2:
                    non_zero = np.max(non_zero, axis=-1)
                pad_counts = np.sum((non_zero == 0), axis=1)
                sort_idx = np.argsort(pad_counts)
                bucket_dict = {k: v.take(sort_idx, axis=0) for k, v in bucket_dict.iteritems()}

            batches_per_bucket = current_bucket_size // batch_size
            if current_bucket_size % batch_size > 0:
                batches_per_bucket += 1

            for batch_num in range(batches_per_bucket):
                # grab the chunk of samples in the current bucket
                batch_start = batch_num * batch_size
                batch_end = batch_start + batch_size
                if batch_start >= current_bucket_size:
                    continue
                batch_end = min(batch_end, current_bucket_size)
                batch_idx = np.arange(batch_start, batch_end)
                batch_dict = {k: v.take(batch_idx, axis=0) for k, v in bucket_dict.iteritems()}
                if trim_function is not None:
                    batch_dict = trim_function(batch_dict)
                if char_level_func is not None:
                    batch_dict = char_level_func(batch_dict, word_id2word, char_word2id)

                if enable_cuda:
                    batch_data = [torch.autograd.Variable(torch.from_numpy(batch_dict[k]).type(torch.LongTensor).cuda()) for k in input_keys + output_keys]
                else:
                    batch_data = [torch.autograd.Variable(torch.from_numpy(batch_dict[k]).type(torch.LongTensor)) for k in input_keys + output_keys]
                yield batch_data


def print_shape_info(dataset):
    for k in dataset:
        print(k, dataset[k].shape)


def print_data_samples(dataset, check_data, head, tail):
    for num in range(head, tail):

        print('#######################################          input_story')
        print(' '.join([dataset.meta_data['id2word'][a] for a in check_data['input_story'][num] if a > 0]))

        print('#######################################          input_question')
        print(' '.join([dataset.meta_data['id2word'][a] for a in check_data['input_question'][num] if a > 0]))

        print('#######################################          answer range')
        a_range = check_data['answer_ranges'][num]
        head, tail = np.argmax(a_range[:, 0]), np.argmax(a_range[:, 1])
        print('#######################################          answer')
        print(' '.join([dataset.meta_data['id2word'][a] for a in check_data['input_story'][num][head: tail + 1] if a > 0]))
        print('#######################################          answerS')
        for ans in check_data['input_answer'][num]:
            print(' '.join([dataset.meta_data['id2word'][a] for a in ans if a > 0]))


def add_char_level_stuff(batch_dict=None, word_id2word=None, char_word2id=None):
    if batch_dict is None or word_id2word is None or char_word2id is None:
        return batch_dict
    # make input_story_char
    word_set = set()
    for input_story in batch_dict['input_story']:
        word_set |= set([word_id2word[w] for w in input_story])  # not pad/bos/eos/unk
    for input_question in batch_dict['input_question']:
        word_set |= set([word_id2word[w] for w in input_question])  # not pad/bos/eos/unk

    word_set = list(word_set)
    max_char_in_word = max(map(len, word_set))

    story_char_matrix = np.zeros((batch_dict['input_story'].shape[0], batch_dict['input_story'].shape[1], max_char_in_word), dtype='int32')
    for i in range(batch_dict['input_story'].shape[0]):
        for j in range(batch_dict['input_story'].shape[1]):
            if batch_dict['input_story'][i][j] == 0:
                continue
            # it's actual word
            _w = word_id2word[batch_dict['input_story'][i][j]]
            _w = _w.lower()
            for k in range(len(_w)):
                try:
                    story_char_matrix[i][j][k] = char_word2id[_w[k]]
                except KeyError:
                    pass
    batch_dict['input_story_char'] = story_char_matrix
    # make target_vocab_pre_EOS char
    question_char_matrix = np.zeros((batch_dict['input_question'].shape[0], batch_dict['input_question'].shape[1], max_char_in_word), dtype='int32')
    for i in range(batch_dict['input_question'].shape[0]):
        for j in range(batch_dict['input_question'].shape[1]):
            if batch_dict['input_question'][i][j] == 0:
                continue
            # it's actual word
            _w = word_id2word[batch_dict['input_question'][i][j]]
            _w = _w.lower()
            for k in range(len(_w)):
                try:
                    question_char_matrix[i][j][k] = char_word2id[_w[k]]
                except KeyError:
                    pass
    batch_dict['input_question_char'] = question_char_matrix

    return batch_dict


def squad_trim(batch_dict):
    batch_dict['input_story'], _margin = trim_batch(batch_dict['input_story'])
    batch_dict['answer_ranges'], _ = trim_batch(batch_dict['answer_ranges'], _margin)
    batch_dict['input_question'], _ = trim_batch(batch_dict['input_question'])
    return batch_dict


def to_str(idx_arrays, lex):
    retval = []
    if 1 == idx_arrays.ndim:
        idx_arrays = np.expand_dims(idx_arrays, 0)
    for g in idx_arrays:
        g = np.trim_zeros(g).tolist()
        g = [item for item in g if item > 1]
        if len(g) > 0:
            g = map(lambda x: lex[x], g)
            retval.append(' '.join(g))
    return retval if 0 < len(retval) else ['']


def htpos2chunks(pred, story):
    # pred: N * 2, head and tail positions
    # stories: N * nws
    head = pred.min()
    tail = pred.max()
    return story[head: tail + 1]


def to_pt(np_matrix, enable_cuda=False):
    if enable_cuda:
        return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
    else:
        return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))


def evaluate(model, data, criterion, trim_function, char_level_func, word_id2word, char_word2id, batch_size=32, enable_cuda=False):
    model.eval()
    data_size = data['input_story'].shape[0]
    total_nll_loss = 0.0
    number_batch = (data_size + batch_size - 1) // batch_size
    exact_match, f1 = 0.0, 0.0

    for i in range(number_batch):

        batch_dict = {'input_story': data['input_story'][i * batch_size: (i + 1) * batch_size],
                      'input_question': data['input_question'][i * batch_size: (i + 1) * batch_size],
                      'answer_ranges': data['answer_ranges'][i * batch_size: (i + 1) * batch_size]}
        batch_dict = trim_function(batch_dict)
        batch_dict = char_level_func(batch_dict, word_id2word, char_word2id)
        gold_standard_answer = data['input_answer'][i * batch_size: (i + 1) * batch_size]

        input_story = batch_dict['input_story']
        preds = model.forward(to_pt(input_story, enable_cuda),
                              to_pt(batch_dict['input_question'], enable_cuda),
                              to_pt(batch_dict['input_story_char'], enable_cuda),
                              to_pt(batch_dict['input_question_char'], enable_cuda))  # batch x time x 2
        # loss
        loss = criterion(preds, to_pt(batch_dict['answer_ranges'], enable_cuda))
        loss = torch.sum(loss).cpu().data.numpy()
        preds = torch.max(preds, 1)[1].cpu().data.numpy().squeeze()  # batch x 2

        for s, p, g in zip(input_story, preds, gold_standard_answer):
            p = htpos2chunks(p, s)
            p = to_str(p, word_id2word)[0]  # one string
            g = to_str(g, word_id2word)  # a list of strings
            exact_match += metric_max_over_ground_truths(
                exact_match_score, p, g)
            f1 += metric_max_over_ground_truths(
                f1_score, p, g)

        total_nll_loss = total_nll_loss + loss

    f1 = float(f1) / float(data_size)
    exact_match = float(exact_match) / float(data_size)
    nll_loss = float(total_nll_loss) / float(data_size)
    return f1, exact_match, nll_loss
# end method evaluate
