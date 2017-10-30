import os
import time
import h5py
import sys
import logging
import argparse
import yaml
import torch
from tqdm import tqdm
from lib.dataset.squad_dataset import SquadDataset
from lib.models.match_lstm import MatchLSTMModel
from lib.objectives.generic import StandardNLL
from lib.utils.setup_logger import setup_logging, log_git_commit
from helpers.generic import print_shape_info, print_data_samples, random_generator, squad_trim, add_char_level_stuff,\
    torch_model_summarize, generator_queue, evaluate

logger = logging.getLogger(__name__)
wait_time = 0.01  # in seconds


def the_main_function(config_dir='config', update_dict=None):
    # read config from yaml file
    cfname = 'config_mlstm.yaml'
    _n = -1
    if args.t:
        # put a _model.yaml in config dir with smaller network
        # cfname = '_' + cfname
        _n = 100

    TRAIN_SIZE = _n
    VALID_SIZE = _n
    TEST_SIZE = _n

    config_file = os.path.join(config_dir, cfname)
    with open(config_file) as reader:
        model_config = yaml.safe_load(reader)
    dataset = SquadDataset(dataset_h5=model_config['dataset']['h5'],
                           data_path='tokenized_squad_v1.1.2/',
                           ignore_case=True)

    train_data, valid_data, test_data = dataset.get_data(train_size=TRAIN_SIZE, valid_size=VALID_SIZE, test_size=TEST_SIZE)
    print_shape_info(train_data)
    if False:
        print('----------------------------------  printing out data shape')
        print_data_samples(dataset, train_data, 12, 15)
        exit(0)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(model_config['scheduling']['cuda_seed'])
    if torch.cuda.is_available():
        if not model_config['scheduling']['enable_cuda']:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(model_config['scheduling']['cuda_seed'])

    _model = MatchLSTMModel(model_config=model_config, data_specs=dataset.meta_data)
    if model_config['scheduling']['enable_cuda']:
        _model.cuda()

    criterion = StandardNLL()
    if model_config['scheduling']['enable_cuda']:
        criterion = criterion.cuda()

    logger.info('finished loading models')
    logger.info(torch_model_summarize(_model))

    # get optimizer / lr
    init_learning_rate = model_config['optimizer']['learning_rate']
    parameters = filter(lambda p: p.requires_grad, _model.parameters())
    if model_config['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif model_config['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    input_keys = ['input_story', 'input_question', 'input_story_char', 'input_question_char']
    output_keys = ['answer_ranges']
    batch_size = model_config['scheduling']['batch_size']
    valid_batch_size = model_config['scheduling']['valid_batch_size']
    _f = h5py.File(dataset.dataset_h5, 'r')
    word_vocab = _f['words_flatten'][0].split('\n')
    word_vocab = list(word_vocab)
    # word2id = dict(zip(word_vocab, range(len(word_vocab))))
    char_vocab = _f['words_flatten_char'][0].split('\n')
    char_vocab = list(char_vocab)
    char_word2id = {}
    for i, ch in enumerate(char_vocab):
        char_word2id[ch] = i

    train_batch_generator = random_generator(data_dict=train_data, batch_size=batch_size,
                                             input_keys=input_keys, output_keys=output_keys,
                                             trim_function=squad_trim, sort_by='input_story',
                                             char_level_func=add_char_level_stuff,
                                             word_id2word=word_vocab, char_word2id=char_word2id,
                                             enable_cuda=model_config['scheduling']['enable_cuda'])
    # train
    number_batch = (train_data['input_story'].shape[0] + batch_size - 1) // batch_size
    data_queue, _ = generator_queue(train_batch_generator, max_q_size=20)
    learning_rate = init_learning_rate
    best_val_f1 = None
    be_patient = 0

    try:
        for epoch in range(model_config['scheduling']['epoch']):
            _model.train()
            sum_loss = 0.0
            with tqdm(total=number_batch, leave=True, ncols=160, ascii=True) as pbar:
                for i in range(number_batch):
                    # qgen train one batch
                    generator_output = None
                    while True:
                        if not data_queue.empty():
                            generator_output = data_queue.get()
                            break
                        else:
                            time.sleep(wait_time)
                    input_story, input_question, input_story_char, input_question_char, answer_ranges = generator_output

                    optimizer.zero_grad()
                    _model.zero_grad()
                    preds = _model.forward(input_story, input_question, input_story_char, input_question_char)  # batch x time x 2
                    # loss
                    loss = criterion(preds, answer_ranges)
                    loss = torch.mean(loss)
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm(_model.parameters(), model_config['optimizer']['clip_grad_norm'])
                    optimizer.step()  # apply gradients
                    preds = torch.max(preds, 1)[1].cpu().data.numpy().squeeze()  # batch x 2
                    batch_loss = loss.cpu().data.numpy()
                    sum_loss += batch_loss * batch_size
                    pbar.set_description('epoch=%d, batch=%d, avg_loss=%.5f, batch_loss=%.5f, lr=%.6f' % (epoch, i, sum_loss / float(batch_size * (i + 1)), batch_loss, learning_rate))
                    pbar.update(1)

            # eval on valid set
            val_f1, val_em, val_nll_loss = evaluate(model=_model, data=valid_data, criterion=criterion,
                                                    trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                                    word_id2word=word_vocab, char_word2id=char_word2id,
                                                    batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
            logger.info("epoch=%d, valid nll loss=%.5f, valid f1=%.5f, valid em=%.5f, lr=%.6f" % (epoch, val_nll_loss, val_f1, val_em, learning_rate))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_f1 or val_f1 > best_val_f1:
                with open(model_config['dataset']['model_save_path'], 'wb') as save_f:
                    torch.save(_model, save_f)
                best_val_f1 = val_f1
                be_patient = 0
            else:
                if epoch >= model_config['optimizer']['learning_rate_decay_from_this_epoch']:
                    if be_patient < model_config['optimizer']['learning_rate_decay_patience']:
                        be_patient += 1
                    else:
                        # Anneal the learning rate if no improvement has been seen in the validation dataset.
                        learning_rate *= model_config['optimizer']['learning_rate_decay_ratio']
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
            logger.info("========================================================================\n")

            test_f1, test_em, test_nll_loss = evaluate(model=_model, data=test_data, criterion=criterion,
                                                       trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                                       word_id2word=word_vocab, char_word2id=char_word2id,
                                                       batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
            logger.info("------------------------------------------------------------------------------------\n")
            logger.info("test: nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))

    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        logger.info('--------------------------------------------\n')
        logger.info('Exiting from training early\n')

    # Load the best saved model.
    with open(model_config['dataset']['model_save_path'], 'rb') as save_f:
        _model = torch.load(save_f)

    # Run on test data.
    logger.info("loading best model------------------------------------------------------------------\n")
    test_f1, test_em, test_nll_loss = evaluate(model=_model, data=test_data, criterion=criterion,
                                               trim_function=squad_trim, char_level_func=add_char_level_stuff,
                                               word_id2word=word_vocab, char_word2id=char_word2id,
                                               batch_size=valid_batch_size, enable_cuda=model_config['scheduling']['enable_cuda'])
    logger.info("------------------------------------------------------------------------------------\n")
    logger.info("nll loss=%.5f, f1=%.5f, em=%.5f" % (test_nll_loss, test_f1, test_em))

    return 0


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding("utf-8")
    for _p in ['logs', 'saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    setup_logging(default_path='config/logging_config.yaml', default_level=logging.INFO, add_time_stamp=True)
    # goal_prompt(logger)
    log_git_commit(logger)
    parser = argparse.ArgumentParser(description="train network.")
    # parser.add_argument("result_file_npy", help="result file with npy format.")  # position argument example.
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-t", action='store_true', help="tiny test")
    args = parser.parse_args()
    the_main_function(config_dir=args.config_dir)
