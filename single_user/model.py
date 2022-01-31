import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, use_pre=False, use_ind=False, indembs=None, induse=None, printfunc=None):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        ninp_mod = ninp
        if use_ind:
            if type(indembs[0]) != type(None):
                assert len(indembs[0][0]) == ninp
            if induse == 'cat':
                ninp_mod = ninp*2

        # modify the size of the input for concatenated embeddings but the output size should be the same as word embedding size; ninp
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp_mod if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp_mod if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp_mod if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.cprint = printfunc
        self.cprint(self.rnns)

        self.decoder = nn.Linear(nhid, ntoken)

        if use_ind:
            self.user_embed = nn.Embedding(ntoken, ninp)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.use_pre = use_pre
        self.use_ind = use_ind
        self.induse = induse
        self.init_weights(indembs)

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self, indembs):
        initrange = 0.1
        if not self.use_pre:
            self.cprint('Initializing encoder weights to random vectors...')
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.weight.data.uniform_(-initrange, initrange)

            if self.use_ind:
                self.user_embed.weight.data.uniform_(-initrange, initrange)
        else:
            self.cprint('Initializing encoder weights to pretrained embeddings...')
            self.encoder.weight.data = torch.FloatTensor(indembs[0])

            # Initialize the individual embedding matrices
            if self.use_ind:
                self.cprint('Initializing user matrices...')
                self.user_embed.weight.data = torch.FloatTensor(indembs[1])

        self.decoder.bias.data.fill_(0)

    def forward(self, input, hidden, users, return_h=False):
        emb, mask = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = emb.cuda()
        emb = self.lockdrop(emb, self.dropouti)

        if self.use_ind:
            uemb = []
            for ik in range(len(users)):
                tuemb = []
                for qk in range(len(users[ik])):
                    uex, _ = embedded_dropout(self.user_embed, input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    tuemb.append(uex)
                uemb.append(torch.stack(tuemb))

            uemb = torch.stack(uemb).cuda()

            if self.induse == 'cat':
                raw_output = torch.cat([emb, uemb], 2)
            elif self.induse == 'sum':
                raw_output = emb + uemb
        else:
            uemb = None
            raw_output = emb

        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        # print('Initializing ' + str(weight))
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
