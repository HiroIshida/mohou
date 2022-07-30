# from https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE
# BSD 3-Clause License
# Copyright (c) 2017,
# All rights reserved.

# from https://github.com/mourga/variational-lstm/blob/master/LICENSE
# MIT License
# Copyright (c) 2021 Katerina Margatina

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import dropout
from torch.nn.parameter import Parameter


class LockedDropout(nn.Module):
    """
    This function applies dropout to the input tensor x.
    The shape of the tensor x in our implementation is (batch_size, seq_len, feature_size)
    (contrary to Merity's AWD that uses (seq_len, batch_size, feature_size)).
    So, we sample a mask from the 'feature_size' dimension,
    but a different mask for each 'batch_size' dimension,
    and expand it in the 'sequence_length' dimension so that
    we apply the SAME mask FOR EACH TIMESTEP of the RNN (= 'seq_len' dim.).

    Code from https://github.com/salesforce/awd-lstm-lm
    paper: https://arxiv.org/pdf/1708.02182.pdf (see Section 4.2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        batch_size, seq_length, feat_size = x.size()
        m = x.data.new(batch_size, 1, feat_size).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=True):
        """
        Dropout class that is paired with a torch module to make sure that the SAME mask
        will be sampled and applied to ALL timesteps.
        :param module: nn. module (e.g. nn.Linear, nn.LSTM)
        :param weights: which weights to apply dropout (names of weights of module)
        :param dropout: dropout to be applied
        :param variational: if True applies Variational Dropout, if False applies DropConnect (different masks!!!)

        Code from https://github.com/salesforce/awd-lstm-lm
        paper: https://arxiv.org/pdf/1708.02182.pdf
        for a difference between dropout and drop connect see https://stats.stackexchange.com/questions/201569/what-is-the-difference-between-dropout-and-drop-connect?fbclid=IwAR26B772sQXXBBpNSI_cu41zIEFu0gVkKgn34E1Wbbb4EbQdkdvmMOgro4s
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        """
        Smerity code I don't understand.
        """
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        """
        This function renames each 'weight name' to 'weight name' + '_raw'
        (e.g. weight_hh_l0 -> weight_hh_l0_raw)
        :return:
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print("Applying weight drop of {} to {}".format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + "_raw", Parameter(w.data))

    def _setweights(self):
        """
        This function samples & applies a dropout mask to the weights of the recurrent layers.
        Specifically, for an LSTM, each gate has
        - a W matrix ('weight_ih') that is multiplied with the input (x_t)
        - a U matrix ('weight_hh') that is multiplied with the previous hidden state (h_t-1)
        We sample a mask (either with Variational Dropout or with DropConnect) and apply it to
        the matrices U and/or W.
        The matrices to be dropped-out are in self.weights.
        A 'weight_hh' matrix is of shape (4*nhidden, nhidden)
        while a 'weight_ih' matrix is of shape (4*nhidden, ninput).

        **** Variational Dropout ****
        With this method, we sample a mask from the tensor (4*nhidden, 1) PER ROW
        and expand it to the full matrix.

        **** DropConnect ****
        With this method, we sample a mask from the tensor (4*nhidden, nhidden) directly
        which means that we apply dropout PER ELEMENT/NEURON.
        :return:
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = None

            mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
            if raw_w.is_cuda:
                mask = mask.cuda()
            mask = dropout(mask, p=self.dropout, training=True)  # type: ignore
            w = mask.expand_as(raw_w) * raw_w
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class VariationalLSTM(nn.Module):
    def __init__(
        self,
        ninput,
        nhidden,
        nlayers=1,
        bidirectional=False,
        dropouti=0.0,
        dropoutw=0.0,
        dropouto=0.0,
        dropout=0.0,
        pack=True,
        last=False,
    ):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        We use batch_first=True for our implementation.
        Tensors are are shape (batch_size, sequence_length, feature_size).
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(VariationalLSTM, self).__init__()

        self.pack = pack
        self.last = last

        self.lockdrop = LockedDropout()

        if not isinstance(nhidden, list):
            nhidden = [nhidden for _ in range(nlayers)]

        self.ninp = ninput
        self.nhid = nhidden
        self.nlayers = nlayers
        self.dropouti = dropouti  # rnn input dropout
        self.dropoutw = dropoutw  # rnn recurrent dropout
        self.dropouto = dropouto  # rnn output dropout
        if dropout == 0.0 and dropouto != 0.0:
            self.dropout = self.dropouto  # rnn output dropout (of the last RNN layer)

        lstms = [
            nn.LSTM(
                input_size=ninput if i == 0 else nhidden[i - 1],
                hidden_size=nhidden[i],
                num_layers=1,
                dropout=0,
                batch_first=True,
            )
            for i in range(nlayers)
        ]

        # Dropout to recurrent layers (matrices weight_hh AND weight_ih of each layer of the RNN)
        dropped_lstms = [
            WeightDrop(rnn, ["weight_hh_l0", "weight_ih_l0"], dropout=dropoutw) for rnn in lstms
        ]

        self.lstms = nn.ModuleList(dropped_lstms)
        # self.init_weights()

    def reorder_hidden(self, hidden, order):
        """
        :param hidden:
        :param order:
        :return:
        """
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]

        return hidden

    def init_hidden(self, bsz):
        """
        Initialise the hidden and cell state (h0, c0) for the first timestep (t=0).
        Both h0, c0 are of shape (num_layers * num_directions, batch_size, hidden_size)
        :param bsz: batch size
        :return:
        """
        weight = next(self.parameters()).data
        return [
            (
                weight.new(1, bsz, self.nhid[i]).zero_(),
                weight.new(1, bsz, self.nhid[i]).zero_(),
            )
            for i in range(self.nlayers)
        ]

    def forward(self, x, hidden=None, lengths=None, return_h=False):
        """

        :param x: tensor of shape (batch_size, seq_len, embedding_size)
        :param hidden: tuple (h0, c0), each of shape (num_layers * num_directions, batch_size, hidden_size)
        :param lengths: tensor (size 1 with true lengths)
        :return:
        """
        batch_size, seq_length, feat_size = x.size()

        # Dropout to inputs of the RNN (dropouti)
        emb = self.lockdrop(x, self.dropouti)

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        raw_output = emb  # input to the first layer
        new_hidden = []
        raw_outputs = []
        outputs = []

        # for each layer of the RNN
        for i, rnn in enumerate(self.lstms):
            # calculate hidden states and output from the l RNN layer
            raw_output, new_h = rnn(raw_output, hidden[i])
            # save them in lists
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.nlayers - 1:
                # apply dropout to the output of the l-th RNN layer (dropouto)
                raw_output = self.lockdrop(raw_output, self.dropouto)
                # save 'dropped-out outputs' in a list
                outputs.append(raw_output)
        hidden = new_hidden

        # Dropout to the output of the last RNN layer (dropout)
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        # result = output.view(output.size(0) * output.size(1), output.size(2))
        result = output
        # result: output of the last RNN layer
        # hidden: hidden state of the last RNN layer
        # raw_outputs: outputs of all RNN layers without dropout
        # outputs: dropped-out outputs of all RNN layers
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden