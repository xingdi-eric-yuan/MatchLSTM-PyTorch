import torch
import numpy as np
import torch.nn.functional as F
from ..utils.embedding_helper import H5EmbeddingManager


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
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
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
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
        if self.enable_cuda:
            word_embedding_init = word_embedding_init.cuda()
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
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias_f = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            self.bias_iog = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        if self.use_layernorm:
            self.layernorm_i = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_h = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_c = LayerNorm(input_dim=self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal(self.weight_hh.data)
        torch.nn.init.xavier_uniform(self.weight_ih.data, gain=1)
        if self.use_bias:
            self.bias_f.data.fill_(1.0)
            self.bias_iog.data.fill_(0.0)

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
        # if (mask_.data == 0).all():
        #     return h_0, c_0
        wh = torch.mm(dropped_h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        if self.use_layernorm:
            wi = self.layernorm_i(wi, mask_)
            wh = self.layernorm_h(wh, mask_)
        pre_act = wi + wh
        if self.use_bias:
            pre_act = pre_act + torch.cat([self.bias_f, self.bias_iog]).unsqueeze(0)

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
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
        super(StackedLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.use_layernorm = use_layernorm
        self.use_highway_connections = use_highway_connections
        self.enable_cuda = enable_cuda
        self.nlayers = len(self.nhids)
        self.stack_rnns()
        if self.use_highway_connections:
            self.build_highway_connections()

    def build_highway_connections(self):
        highway_connections_x = [torch.nn.Linear(self.nemb if i == 0 else self.nhids[i - 1], self.nemb if i == 0 else self.nhids[i - 1]) for i in range(self.nlayers)]
        highway_connections_x_x = [torch.nn.Linear(self.nemb if i == 0 else self.nhids[i - 1], self.nhids[i], bias=False) for i in range(self.nlayers)]
        highway_connections_h = [torch.nn.Linear(self.nemb if i == 0 else self.nhids[i - 1], self.nhids[i]) for i in range(self.nlayers)]
        self.highway_connections_x = torch.nn.ModuleList(highway_connections_x)
        self.highway_connections_x_x = torch.nn.ModuleList(highway_connections_x_x)
        self.highway_connections_h = torch.nn.ModuleList(highway_connections_h)
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform(self.highway_connections_x[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.highway_connections_h[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.highway_connections_x_x[i].weight.data, gain=1)
            self.highway_connections_x[i].bias.data.fill_(0)
            self.highway_connections_h[i].bias.data.fill_(0)

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
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda(),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda())]
                    for i in range(self.nlayers)]
        else:
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()))]
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
        state_stp = self.get_init_hidden(x.size(0))
        hidden_to_hidden_dropout_masks = [None for _ in range(self.nlayers)]

        for d, rnn in enumerate(self.rnns):
            for t in range(x.size(1)):

                input_mask = mask[:, t]
                if d == 0:
                    # 0th layer
                    curr_input = x[:, t]
                else:
                    curr_input = state_stp[d - 1][t][0]
                # apply dropout layer-to-layer
                drop_input = F.dropout(curr_input, p=self.dropout_between_rnn_layers, training=self.training) if d > 0 else curr_input
                previous_h, previous_c = state_stp[d][t]
                if t == 0:
                    # get hidden to hidden dropout mask at 0th time step of each rnn layer, and freeze them at teach time step
                    hidden_to_hidden_dropout_masks[d] = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
                dropped_previous_h = hidden_to_hidden_dropout_masks[d] * previous_h

                new_h, new_c = rnn.forward(drop_input, input_mask, previous_h, previous_c, dropped_previous_h)
                state_stp[d].append((new_h, new_c))

            if self.use_highway_connections:
                for t in range(x.size(1)):
                    input_mask = mask[:, t]
                    if d == 0:
                        # 0th layer
                        curr_input = x[:, t]
                    else:
                        curr_input = state_stp[d - 1][t][0]
                    new_h, new_c = state_stp[d][t]
                    gate_x = F.sigmoid(self.highway_connections_x[d].forward(curr_input))
                    gate_h = F.sigmoid(self.highway_connections_h[d].forward(curr_input))
                    new_h = self.highway_connections_x_x[d].forward(curr_input * gate_x) + gate_h * new_h  # batch x hid
                    new_h = new_h * input_mask.unsqueeze(1)
                    state_stp[d][t] = (new_h, new_c)

        states = [h[0] for h in state_stp[-1][1:]]  # list of batch x hid
        states = torch.stack(states, 1)  # batch x time x hid
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
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
        super(UniLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.rnn = StackedLSTM(nemb=nemb,
                               nhids=nhids,
                               dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                               dropout_between_rnn_layers=dropout_between_rnn_layers,
                               dropout_in_rnn_weights=dropout_in_rnn_weights,
                               use_layernorm=use_layernorm,
                               use_highway_connections=use_highway_connections,
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
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
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
                                       use_highway_connections=use_highway_connections,
                                       enable_cuda=enable_cuda
                                       )

        self.backward_rnn = StackedLSTM(nemb=self.nemb,
                                        nhids=[hid // 2 for hid in self.nhids],
                                        dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                        dropout_between_rnn_layers=dropout_between_rnn_layers,
                                        dropout_in_rnn_weights=dropout_in_rnn_weights,
                                        use_layernorm=use_layernorm,
                                        use_highway_connections=use_highway_connections,
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
        concat_states = concat_states * mask.unsqueeze(-1)  # batch x time x hid
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
        self.init_weights()

    def init_weights(self):
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform(self.W_p[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.W_q[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.W_r[i].weight.data, gain=1)
            self.W_p[i].bias.data.fill_(0)
            self.W_q[i].bias.data.fill_(0)
            self.W_r[i].bias.data.fill_(0)
            torch.nn.init.normal(self.w[i].data, mean=0, std=0.05)
            self.match_b[i].data.fill_(1.0)

    def forward(self, input_p, mask_p, input_q, mask_q, h_tm1, depth):
        G_p = self.W_p[depth](input_p).unsqueeze(1)  # batch x None x out
        G_q = self.W_q[depth](input_q)  # batch x time x out
        G_r = self.W_r[depth](h_tm1).unsqueeze(1)  # batch x None x out
        G = F.tanh(G_p + G_q + G_r)  # batch x time x out
        alpha = torch.matmul(G, self.w[depth])  # batch x time
        alpha = alpha + self.match_b[depth].unsqueeze(0)  # batch x time
        alpha = masked_softmax(alpha, mask_q, axis=-1)  # batch x time
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
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
        super(StackedMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = attention_layer
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.use_layernorm = use_layernorm
        self.use_highway_connections = use_highway_connections
        self.enable_cuda = enable_cuda
        self.nlayers = len(self.nhids)
        self.stack_rnns()
        if self.use_highway_connections:
            self.build_highway_connections()

    def build_highway_connections(self):
        highway_connections_x = [torch.nn.Linear(self.input_p_dim if i == 0 else self.nhids[i - 1], self.input_p_dim if i == 0 else self.nhids[i - 1]) for i in range(self.nlayers)]
        highway_connections_x_x = [torch.nn.Linear(self.input_p_dim if i == 0 else self.nhids[i - 1], self.nhids[i], bias=False) for i in range(self.nlayers)]
        highway_connections_h = [torch.nn.Linear(self.input_p_dim if i == 0 else self.nhids[i - 1], self.nhids[i]) for i in range(self.nlayers)]
        self.highway_connections_x = torch.nn.ModuleList(highway_connections_x)
        self.highway_connections_x_x = torch.nn.ModuleList(highway_connections_x_x)
        self.highway_connections_h = torch.nn.ModuleList(highway_connections_h)
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform(self.highway_connections_x[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.highway_connections_h[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.highway_connections_x_x[i].weight.data, gain=1)
            self.highway_connections_x[i].bias.data.fill_(0)
            self.highway_connections_h[i].bias.data.fill_(0)

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
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda(),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda())]
                    for i in range(self.nlayers)]
        else:
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()))]
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
        state_stp = self.get_init_hidden(batch_size)

        for d, rnn in enumerate(self.rnns):
            for t in range(input_p.size(1)):

                input_mask = mask_p[:, t]
                if d == 0:
                    # 0th layer
                    curr_input = input_p[:, t]
                else:
                    curr_input = state_stp[d - 1][t][0]
                # apply dropout layer-to-layer
                drop_input = F.dropout(curr_input, p=self.dropout_between_rnn_layers, training=self.training) if d > 0 else curr_input
                previous_h, previous_c = state_stp[d][t]
                if t == 0:
                    # get hidden to hidden dropout mask at 0th time step of each rnn layer, and freeze them at teach time step
                    hidden_to_hidden_dropout_masks[d] = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
                dropped_previous_h = hidden_to_hidden_dropout_masks[d] * previous_h

                drop_input = self.attention_layer.forward(drop_input, input_mask, input_q, mask_q, h_tm1=dropped_previous_h, depth=d)
                new_h, new_c = rnn.forward(drop_input, input_mask, previous_h, previous_c, dropped_previous_h)
                state_stp[d].append((new_h, new_c))

            if self.use_highway_connections:
                for t in range(input_p.size(1)):
                    input_mask = mask_p[:, t]
                    if d == 0:
                        # 0th layer
                        curr_input = input_p[:, t]
                    else:
                        curr_input = state_stp[d - 1][t][0]
                    new_h, new_c = state_stp[d][t]
                    gate_x = F.sigmoid(self.highway_connections_x[d].forward(curr_input))
                    gate_h = F.sigmoid(self.highway_connections_h[d].forward(curr_input))
                    new_h = self.highway_connections_x_x[d].forward(curr_input * gate_x) + gate_h * new_h  # batch x hid
                    new_h = new_h * input_mask.unsqueeze(1)
                    state_stp[d][t] = (new_h, new_c)

        states = [h[0] for h in state_stp[-1][1:]]  # list of batch x hid
        states = torch.stack(states, 1)  # batch x time x hid
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
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
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
                                    use_highway_connections=use_highway_connections,
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
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
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
                                            use_highway_connections=use_highway_connections,
                                            enable_cuda=enable_cuda)

        self.backward_rnn = StackedMatchLSTM(input_p_dim=self.input_p_dim,
                                             input_q_dim=self.input_q_dim,
                                             nhids=[hid // 2 for hid in self.nhids],
                                             attention_layer=self.attention_layer,
                                             dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                             dropout_between_rnn_layers=dropout_between_rnn_layers,
                                             dropout_in_rnn_weights=dropout_in_rnn_weights,
                                             use_layernorm=use_layernorm,
                                             use_highway_connections=use_highway_connections,
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

        # backward pass
        input_p_inverted = self.flip(input_p, flip_dim=1)  # batch x time x p_dim (backward)
        mask_p_inverted = self.flip(mask_p, flip_dim=1)  # batch x time (backward)
        backward_states, _ = self.backward_rnn.forward(input_p_inverted, mask_p_inverted, input_q, mask_q)
        backward_last_state = backward_states[:, -1]  # batch x hid/2
        backward_states = self.flip(backward_states, flip_dim=1)  # batch x time x hid/2

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid
        concat_states = concat_states * mask_p.unsqueeze(-1)  # batch x time x hid
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
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.V.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_a.weight.data, gain=1)
        self.V.bias.data.fill_(0)
        self.W_a.bias.data.fill_(0)
        torch.nn.init.normal(self.v.data, mean=0, std=0.05)
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
        beta = masked_softmax(beta, mask_r, axis=-1)  # batch x time
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
        res = torch.stack(beta_list, 2)  # batch x time x 2
        res = res * x_mask.unsqueeze(2)  # batch x time x 2
        return res


class FastBiLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastBiLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = [h // 2 for h in nhids]
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1] * 2,
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=True) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = torch.cat([last_state[0], last_state[1]], 1)  # batch x hid_f+hid_b

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask
