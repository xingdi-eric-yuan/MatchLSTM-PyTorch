import torch
import numpy as np
import torch.nn.functional as F
from ..utils.embedding_helper import H5EmbeddingManager


def masked_softmax(x, m=None, axis=-1, enable_cuda=False):
    '''
    Softmax with mask (optional)
    '''
    fifteen = torch.autograd.Variable(torch.ones(x.size())) * 15.0
    minus_fifteen = torch.autograd.Variable(torch.ones(x.size())) * -15.0
    if enable_cuda:
        fifteen = fifteen.cuda()
        minus_fifteen = minus_fifteen.cuda()
    x = torch.max(x, minus_fifteen)
    x = torch.min(x, fifteen)
    if m is not None:
        m = m.type(torch.FloatTensor)
        if enable_cuda:
            m = m.cuda()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


class LayerNorm(torch.nn.Module):

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(input_dim))
        self.beta = torch.nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x, mask):
        # x:        nbatch x hidden
        # mask:     nbatch
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output * mask.unsqueeze(1)


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, trainable, id2word=None,
                 word_dropout_rate=0.0, dropout_rate=0.0, embedding_oov_init='random',
                 embedding_type='random', pretrained_embedding_path=None,
                 enable_cuda=False):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.id2word = id2word
        self.embedding_type = embedding_type
        self.embedding_oov_init = embedding_oov_init
        self.pretrained_embedding_path = pretrained_embedding_path
        self.word_dropout_rate = word_dropout_rate
        self.trainable = trainable
        self.enable_cuda = enable_cuda
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def embedding_init(self):
        # Embeddings
        if self.embedding_type == 'random':
            W_e = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
            W_e[0, :] = 0
            word_embedding_init = W_e
        else:
            embedding_initr = H5EmbeddingManager(self.pretrained_embedding_path)
            word_embedding_init = embedding_initr.word_embedding_initialize(self.id2word,
                                                                            dim_size=self.embedding_size,
                                                                            mode=self.embedding_type,
                                                                            oov_init=self.embedding_oov_init)
            del embedding_initr
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        return mask

    def embedded_dropout(self, words, scale=None):
        dropout = self.word_dropout_rate if self.training else 0.0
        if dropout > 0.:
            mask = self.embedding_layer.weight.data.new().resize_((
                self.embedding_layer.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(self.embedding_layer.weight) / (1 - dropout)
            mask = torch.autograd.Variable(mask)
            if self.enable_cuda:
                mask = mask.cuda()
            masked_embed_weight = mask * self.embedding_layer.weight
        else:
            masked_embed_weight = self.embedding_layer.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
        padding_idx = self.embedding_layer.padding_idx
        if padding_idx is None:
            padding_idx = -1
        X = self.embedding_layer._backend.Embedding.apply(
            words, masked_embed_weight,
            padding_idx, self.embedding_layer.max_norm, self.embedding_layer.norm_type,
            self.embedding_layer.scale_grad_by_freq, self.embedding_layer.sparse)
        return X

    def forward(self, x):
        # drop entire word embeddings
        embeddings = self.embedded_dropout(x)
        # apply standard dropout
        embeddings = self.dropout(embeddings)  # batch x time x emb
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False, enable_cuda=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', torch.nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class LSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_layernorm=False, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.weight_ih = torch.nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        if self.use_layernorm:
            self.layernorm_i = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_h = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_c = LayerNorm(input_dim=self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input_, mask_, h_0, c_0, dropped_h_0):
        """
        Args:
            input_:     A (batch, input_size) tensor containing input features.
            mask_:      (batch)
            hx:         A tuple (h_0, c_0), which contains the initial hidden
                        and cell state, where the size of both states is
                        (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        wh = torch.mm(dropped_h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        if self.use_layernorm:
            wi = self.layernorm_i(wi, mask_)
            wh = self.layernorm_h(wh, mask_)
        pre_act = wi + wh
        if self.use_bias:
            pre_act = pre_act + self.bias.unsqueeze(0)

        f, i, o, g = torch.split(pre_act, split_size=self.hidden_size, dim=1)
        expand_mask_ = mask_.unsqueeze(1)  # batch x None
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1 = c_1 * expand_mask_ + c_0 * (1 - expand_mask_)
        if self.use_layernorm:
            h_1 = torch.sigmoid(o) * torch.tanh(self.layernorm_c(c_1, mask_))
        else:
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        h_1 = h_1 * expand_mask_ + h_0 * (1 - expand_mask_)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class StackedLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:
            encoding:   batch x time x h
            mask:       batch x time
    Dropout types:
        dropout_between_rnn_hiddens -- across time step
        dropout_between_rnn_layers -- if multi layer rnns
        dropout_in_rnn_weights -- rnn weight dropout
    '''

    def __init__(self, nemb, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, enable_cuda=False):
        super(StackedLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.use_layernorm = use_layernorm
        self.enable_cuda = enable_cuda
        self.nlayers = len(self.nhids)
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [LSTMCell(self.nemb if i == 0 else self.nhids[i - 1], self.nhids[i], use_layernorm=self.use_layernorm, use_bias=True)
                for i in range(self.nlayers)]
        if self.dropout_in_rnn_weights > 0.:
            print('Applying hidden weight dropout {:.2f}'.format(self.dropout_in_rnn_weights))
            rnns = [WeightDrop(rnn, ['weight_hh'], dropout=self.dropout_in_rnn_weights)
                    for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.enable_cuda:
            return [(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda(),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda())
                    for i in range(self.nlayers)]
        else:
            return [(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()))
                    for i in range(self.nlayers)]

    def get_dropout_mask(self, x, _rate=0.5):
        mask = torch.ones(x.size())
        if self.training and _rate > 0.05:
            mask = mask.bernoulli_(1 - _rate) / (1 - _rate)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        if self.enable_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x, mask):
        state_stp = [self.get_init_hidden(x.size(0))]
        hidden_to_hidden_dropout_masks = [None for _ in range(self.nlayers)]
        for t in range(x.size(1)):
            state_depth = []
            input_mask = mask[:, t]

            for d, rnn in enumerate(self.rnns):
                if d == 0:
                    # 0th layer
                    curr_input = x[:, t]
                else:
                    curr_input = state_stp[t][d - 1][0]
                # apply dropout layer-to-layer
                drop_input = F.dropout(curr_input, p=self.dropout_between_rnn_layers, training=self.training) if d > 0 else curr_input
                previous_h, previous_c = state_stp[t][d]
                if t == 0:
                    # get hidden to hidden dropout mask at 0th time step of each rnn layer, and freeze them at teach time step
                    hidden_to_hidden_dropout_masks[d] = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
                dropped_previous_h = hidden_to_hidden_dropout_masks[d] * previous_h
                new_h, new_c = rnn(drop_input, input_mask, previous_h, previous_c, dropped_previous_h)
                state_depth.append((new_h, new_c))

            state_stp.append(state_depth)

        states = [h[-1][0].unsqueeze(1) for h in state_stp[1:]]  # list of batch x 1 x hid
        states = torch.cat(states, 1)  # batch x time x hid
        return states, mask


class UniLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:encoding:   batch x time x hid
            last state: batch x hid
            mask:       batch x time
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, nemb, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, enable_cuda=False):
        super(UniLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.rnn = StackedLSTM(nemb=nemb,
                               nhids=nhids,
                               dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                               dropout_between_rnn_layers=dropout_between_rnn_layers,
                               dropout_in_rnn_weights=dropout_in_rnn_weights,
                               use_layernorm=use_layernorm,
                               enable_cuda=enable_cuda
                               )

    def forward(self, x, mask):
        # stacked rnn
        states, _ = self.rnn.forward(x, mask)
        last_state = states[:, -1]  # batch x hid
        states = states * mask.unsqueeze(-1)  # batch x time x hid
        return states, last_state, mask


class BiLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:encoding:   batch x time x hid
            last state: batch x hid
            mask:       batch x time
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, nemb, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, enable_cuda=False):
        super(BiLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.enable_cuda = enable_cuda
        self.forward_rnn = StackedLSTM(nemb=self.nemb,
                                       nhids=[hid // 2 for hid in self.nhids],
                                       dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                       dropout_between_rnn_layers=dropout_between_rnn_layers,
                                       dropout_in_rnn_weights=dropout_in_rnn_weights,
                                       use_layernorm=use_layernorm,
                                       enable_cuda=enable_cuda
                                       )

        self.backward_rnn = StackedLSTM(nemb=self.nemb,
                                        nhids=[hid // 2 for hid in self.nhids],
                                        dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                        dropout_between_rnn_layers=dropout_between_rnn_layers,
                                        dropout_in_rnn_weights=dropout_in_rnn_weights,
                                        use_layernorm=use_layernorm,
                                        enable_cuda=enable_cuda
                                        )

    def flip(self, tensor, flip_dim=0):
        # flip
        idx = [i for i in range(tensor.size(flip_dim) - 1, -1, -1)]
        idx = torch.autograd.Variable(torch.LongTensor(idx))
        if self.enable_cuda:
            idx = idx.cuda()
        inverted_tensor = tensor.index_select(flip_dim, idx)
        return inverted_tensor

    def forward(self, x, mask):

        embeddings = x
        embeddings_inverted = self.flip(embeddings, flip_dim=1)  # batch x time x emb (backward)
        mask_inverted = self.flip(mask, flip_dim=1)  # batch x time (backward)

        forward_states, _ = self.forward_rnn.forward(embeddings, mask)  # batch x time x hid/2
        forward_last_state = forward_states[:, -1]  # batch x hid/2

        backward_states, _ = self.backward_rnn.forward(embeddings_inverted, mask_inverted)  # batch x time x hid/2 (backward)
        backward_last_state = backward_states[:, -1]  # batch x hid/2
        backward_states = self.flip(backward_states, flip_dim=1)  # batch x time x hid/2

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid
        concat_last_state = torch.cat([forward_last_state, backward_last_state], -1)  # batch x hid
        return concat_states, concat_last_state, mask


class TimeDistributedRNN(torch.nn.Module):
    '''
    input:  embedding:  batch x T x time x emb
            mask:       batch x T x time
    output: sequence:   batch x T x time x enc
            last state: batch x T x enc
            mask:       batch x T x time
    '''

    def __init__(self, rnn):
        super(TimeDistributedRNN, self).__init__()
        self.rnn = rnn

    def forward(self, x, mask):

        batch_size, T, time = x.size(0), x.size(1), x.size(2)
        x = x.view(batch_size * T, time, -1)  # batch*T x time x emb
        _mask = mask.view(batch_size * T, time)  # batch*T x time

        seq, last, _ = self.rnn.forward(x, _mask)
        seq = seq.view(batch_size, T, time, -1)  # batch x T x time x enc
        last = last.view(batch_size, T, -1)  # batch x T x enc
        return seq, last, mask


class TimeDistributedEmbedding(torch.nn.Module):
    '''
    input:  embedding:  batch x T x time
    output: sequence:   batch x T x time x emb
            mask:       batch x T x time
    '''

    def __init__(self, emb_layer):
        super(TimeDistributedEmbedding, self).__init__()
        self.emb_layer = emb_layer

    def forward(self, x):
        batch_size, T, time = x.size(0), x.size(1), x.size(2)
        x = x.view(-1, time)  # batch*T x time
        emb, mask = self.emb_layer.forward(x)
        emb = emb.view(batch_size, T, time, -1)
        mask = mask.view(batch_size, T, time)
        return emb, mask


class MatchLSTMAttention(torch.nn.Module):
    '''
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
                depth:      int
        output: z:          batch x inp_p+inp_q
    '''

    def __init__(self, input_p_dim, input_q_dim, output_dim, enable_cuda=False):
        super(MatchLSTMAttention, self).__init__()
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.enable_cuda = enable_cuda
        self.nlayers = len(self.output_dim)

        W_p = [torch.nn.Linear(self.input_p_dim if i == 0 else self.output_dim[i - 1], self.output_dim[i]) for i in range(self.nlayers)]
        W_q = [torch.nn.Linear(self.input_q_dim, self.output_dim[i]) for i in range(self.nlayers)]
        W_r = [torch.nn.Linear(self.output_dim[i], self.output_dim[i]) for i in range(self.nlayers)]
        w = [torch.nn.Parameter(torch.FloatTensor(self.output_dim[i])) for i in range(self.nlayers)]
        match_b = [torch.nn.Parameter(torch.FloatTensor(1)) for i in range(self.nlayers)]

        self.W_p = torch.nn.ModuleList(W_p)
        self.W_q = torch.nn.ModuleList(W_q)
        self.W_r = torch.nn.ModuleList(W_r)
        self.w = torch.nn.ParameterList(w)
        self.match_b = torch.nn.ParameterList(match_b)
        self._eps = 1e-6
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for i in range(self.nlayers):
            self.W_p[i].weight.data.uniform_(-initrange, initrange)
            self.W_p[i].bias.data.fill_(0)
            self.W_q[i].weight.data.uniform_(-initrange, initrange)
            self.W_q[i].bias.data.fill_(0)
            self.W_r[i].weight.data.uniform_(-initrange, initrange)
            self.W_r[i].bias.data.fill_(0)
            self.w[i].data.uniform_(-0.05, 0.05)
            self.match_b[i].data.fill_(1.0)

    def forward(self, input_p, mask_p, input_q, mask_q, h_tm1, depth):
        G_p = self.W_p[depth](input_p).unsqueeze(1)  # batch x None x out
        G_q = self.W_q[depth](input_q)  # batch x time x out
        G_r = self.W_r[depth](h_tm1).unsqueeze(1)  # batch x None x out
        G = F.tanh(G_p + G_q + G_r)  # batch x time x out
        alpha = torch.matmul(G, self.w[depth])  # batch x time
        alpha = alpha + self.match_b[depth].unsqueeze(0)  # batch x time
        alpha = masked_softmax(alpha, mask_q, axis=-1, enable_cuda=self.enable_cuda)  # batch x time
        alpha = alpha.unsqueeze(1)  # batch x 1 x time
        # batch x time x input_q, batch x 1 x time
        z = torch.bmm(alpha, input_q)  # batch x 1 x input_q
        z = z.squeeze(1)  # batch x input_q
        z = torch.cat([input_p, z], 1)  # batch x input_p+input_q
        return z


class StackedMatchLSTM(torch.nn.Module):
    '''
    inputs: p:          batch x time x inp_p
            mask_p:     batch x time
            q:          batch x time x inp_q
            mask_q:     batch x time
    outputs:
            encoding:   batch x time x h
            mask_p:     batch x time
    Dropout types:
        dropout_between_rnn_hiddens -- across time step
        dropout_between_rnn_layers -- if multi layer rnns
        dropout_in_rnn_weights -- rnn weight dropout
    '''

    def __init__(self, input_p_dim, input_q_dim, nhids, attention_layer,
                 dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, enable_cuda=False):
        super(StackedMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = attention_layer
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.use_layernorm = use_layernorm
        self.enable_cuda = enable_cuda
        self.nlayers = len(self.nhids)
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [LSTMCell((self.input_p_dim + self.input_q_dim) if i == 0 else (self.nhids[i - 1] + self.input_q_dim), self.nhids[i], use_layernorm=self.use_layernorm, use_bias=True)
                for i in range(self.nlayers)]
        if self.dropout_in_rnn_weights > 0.:
            print('Applying hidden weight dropout {:.2f}'.format(self.dropout_in_rnn_weights))
            rnns = [WeightDrop(rnn, ['weight_hh'], dropout=self.dropout_in_rnn_weights)
                    for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.enable_cuda:
            return [(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda(),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda())
                    for i in range(self.nlayers)]
        else:
            return [(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()))
                    for i in range(self.nlayers)]

    def get_dropout_mask(self, x, _rate=0.5):
        mask = torch.ones(x.size())
        if self.training and _rate > 0.05:
            mask = mask.bernoulli_(1 - _rate) / (1 - _rate)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        if self.enable_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, input_p, mask_p, input_q, mask_q):
        batch_size = input_p.size(0)
        hidden_to_hidden_dropout_masks = [None for _ in range(self.nlayers)]
        state_stp = [self.get_init_hidden(batch_size)]
        for t in range(input_p.size(1)):
            state_depth = []
            input_mask = mask_p[:, t]

            for d, rnn in enumerate(self.rnns):
                if d == 0:
                    # 0th layer
                    curr_input = input_p[:, t]
                else:
                    curr_input = state_stp[t][d - 1][0]
                # apply dropout layer-to-layer
                drop_input = F.dropout(curr_input, p=self.dropout_between_rnn_layers, training=self.training) if d > 0 else curr_input
                previous_h, previous_c = state_stp[t][d]
                if t == 0:
                    # get hidden to hidden dropout mask at 0th time step of each rnn layer, and freeze them at teach time step
                    hidden_to_hidden_dropout_masks[d] = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
                dropped_previous_h = hidden_to_hidden_dropout_masks[d] * previous_h

                drop_input = self.attention_layer.forward(drop_input, input_mask, input_q, mask_q, h_tm1=dropped_previous_h, depth=d)
                new_h, new_c = rnn(drop_input, input_mask, previous_h, previous_c, dropped_previous_h)
                state_depth.append((new_h, new_c))

            state_stp.append(state_depth)

        states = [h[-1][0].unsqueeze(1) for h in state_stp[1:]]  # list of batch x 1 x hid
        states = torch.cat(states, 1)  # batch x time x hid
        return states, mask_p


class UniMatchLSTM(torch.nn.Module):
    '''
    inputs: p:          batch x time x inp_p
            mask_p:     batch x time
            q:          batch x time x inp_q
            mask_q:     batch x time
    outputs:encoding:   batch x time x hid
            last state: batch x hid
            mask_p:     batch x time
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, input_p_dim, input_q_dim, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, enable_cuda=False):
        super(UniMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = MatchLSTMAttention(input_p_dim, input_q_dim,
                                                  output_dim=nhids,
                                                  enable_cuda=enable_cuda)

        self.rnn = StackedMatchLSTM(input_p_dim=self.input_p_dim,
                                    input_q_dim=self.input_q_dim,
                                    nhids=self.nhids,
                                    attention_layer=self.attention_layer,
                                    dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                    dropout_between_rnn_layers=dropout_between_rnn_layers,
                                    dropout_in_rnn_weights=dropout_in_rnn_weights,
                                    use_layernorm=use_layernorm,
                                    enable_cuda=enable_cuda)

    def forward(self, input_p, mask_p, input_q, mask_q):
        # stacked rnn
        states, _ = self.rnn.forward(input_p, mask_p, input_q, mask_q)
        last_state = states[:, -1]  # batch x hid
        states = states * mask_p.unsqueeze(-1)  # batch x time x hid
        return states, last_state, mask_p


class BiMatchLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:encoding:   batch x time x hid
            last state: batch x hid
            mask:       batch x time
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, input_p_dim, input_q_dim, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, enable_cuda=False):
        super(BiMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.enable_cuda = enable_cuda
        self.attention_layer = MatchLSTMAttention(input_p_dim, input_q_dim,
                                                  output_dim=[hid // 2 for hid in self.nhids],
                                                  enable_cuda=enable_cuda)

        self.forward_rnn = StackedMatchLSTM(input_p_dim=self.input_p_dim,
                                            input_q_dim=self.input_q_dim,
                                            nhids=[hid // 2 for hid in self.nhids],
                                            attention_layer=self.attention_layer,
                                            dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                            dropout_between_rnn_layers=dropout_between_rnn_layers,
                                            dropout_in_rnn_weights=dropout_in_rnn_weights,
                                            use_layernorm=use_layernorm,
                                            enable_cuda=enable_cuda)

        self.backward_rnn = StackedMatchLSTM(input_p_dim=self.input_p_dim,
                                             input_q_dim=self.input_q_dim,
                                             nhids=[hid // 2 for hid in self.nhids],
                                             attention_layer=self.attention_layer,
                                             dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                             dropout_between_rnn_layers=dropout_between_rnn_layers,
                                             dropout_in_rnn_weights=dropout_in_rnn_weights,
                                             use_layernorm=use_layernorm,
                                             enable_cuda=enable_cuda)

    def flip(self, tensor, flip_dim=0):
        # flip
        idx = [i for i in range(tensor.size(flip_dim) - 1, -1, -1)]
        idx = torch.autograd.Variable(torch.LongTensor(idx))
        if self.enable_cuda:
            idx = idx.cuda()
        inverted_tensor = tensor.index_select(flip_dim, idx)
        return inverted_tensor

    def forward(self, input_p, mask_p, input_q, mask_q):

        # forward pass
        forward_states, _ = self.forward_rnn.forward(input_p, mask_p, input_q, mask_q)
        forward_last_state = forward_states[:, -1]  # batch x hid/2
        forward_states = forward_states * mask_p.unsqueeze(-1)  # batch x time x hid/2

        # backward pass
        input_p_inverted = self.flip(input_p, flip_dim=1)  # batch x time x p_dim (backward)
        mask_p_inverted = self.flip(mask_p, flip_dim=1)  # batch x time (backward)
        backward_states, _ = self.backward_rnn.forward(input_p_inverted, mask_p_inverted, input_q, mask_q)
        backward_last_state = backward_states[:, -1]  # batch x hid/2
        backward_states = backward_states * mask_p_inverted.unsqueeze(-1)  # batch x time x hid/2
        backward_states = self.flip(backward_states, flip_dim=1)  # batch x time x hid/2

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid
        concat_last_state = torch.cat([forward_last_state, backward_last_state], -1)  # batch x hid

        return concat_states, concat_last_state, mask_p


class BoundaryDecoderAttention(torch.nn.Module):
    '''
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
                depth:      int
        output: z:          batch x inp_p+inp_q
    '''

    def __init__(self, input_dim, output_dim, enable_cuda=False):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enable_cuda = enable_cuda

        self.V = torch.nn.Linear(self.input_dim, self.output_dim)
        self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self._eps = 1e-6
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.V.weight.data.uniform_(-initrange, initrange)
        self.V.bias.data.fill_(0)
        self.W_a.weight.data.uniform_(-initrange, initrange)
        self.W_a.bias.data.fill_(0)
        self.v.data.uniform_(-0.05, 0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r, mask_r, h_tm1):
        # H_r: batch x time x inp
        # mask_r: batch x time
        # h_tm1: batch x out
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V.forward(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a.forward(h_tm1)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))  # batch x time x out

        beta = torch.matmul(Fk, self.v)  # batch x time
        beta = beta + self.c.unsqueeze(0)  # batch x time
        beta = masked_softmax(beta, mask_r, axis=-1, enable_cuda=self.enable_cuda)  # batch x time
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        return z, beta


class BoundaryDecoder(torch.nn.Module):
    '''
    input:  encoded stories:    batch x time x input_dim
            story mask:         batch x time
            init states:        batch x hid
    '''

    def __init__(self, input_dim, hidden_dim, enable_cuda=False):
        super(BoundaryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enable_cuda = enable_cuda
        self.attention_layer = BoundaryDecoderAttention(input_dim=input_dim,
                                                        output_dim=hidden_dim,
                                                        enable_cuda=enable_cuda)

        self.rnn = LSTMCell(self.input_dim, self.hidden_dim, use_layernorm=False, use_bias=True)

    def forward(self, x, x_mask, h_0):

        state_stp = [(h_0, h_0)]
        beta_list = []
        if self.enable_cuda:
            mask = torch.autograd.Variable(torch.ones(x.size(0)).cuda())  # fake mask
        else:
            mask = torch.autograd.Variable(torch.ones(x.size(0)))  # fake mask
        for t in range(2):

            previous_h, previous_c = state_stp[t]
            curr_input, beta = self.attention_layer.forward(x, x_mask, h_tm1=previous_h)
            new_h, new_c = self.rnn(curr_input, mask, previous_h, previous_c, previous_h)
            state_stp.append((new_h, new_c))
            beta_list.append(beta)

        # beta list: list of batch x time
        res = [b.unsqueeze(2) for b in beta_list]  # list of batch x time x 1
        res = torch.cat(res, 2)  # batch x time x 2
        res = res * x_mask.unsqueeze(2)  # batch x time x 2
        return res
