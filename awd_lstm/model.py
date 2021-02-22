import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

from utils import DEMOVARS

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, use_pre=False, use_demo=False, useone=None, demoembs=None, demouse=None, mainmatrix=0, printfunc=None):#, match_input_size=False
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # self.match_input_size = match_input_size

        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        ninp_mod = ninp # make this bigger if we want to concatenate demographic embeddings
        if use_demo:
            assert len(demoembs[0][0][1]) == ninp
            if demouse == 'cat':
                if useone != None:
                    ninp_mod = ninp*2
                else:
                    ninp_mod = ninp*5

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
        # self.age_decode = nn.Linear(nhid, len(demoembs[0]))
        # self.location_decode = nn.Linear(nhid, len(demoembs[1]))
        # self.religion_decode = nn.Linear(nhid, len(demoembs[2]))
        # self.gender_decode = nn.Linear(nhid, len(demoembs[3]))

        if useone != None:
            self.cprint('Using one demographic input: ' + str(useone))
        elif use_demo:
            self.cprint('Using all four demographic inputs')

        if use_demo and useone in ['age', None]:
            self.age_embed = torch.nn.ModuleList([nn.Embedding(ntoken, ninp) for i in DEMOVARS['AGE']])
        if use_demo and useone in ['location', None]:
            self.location_embed = torch.nn.ModuleList([nn.Embedding(ntoken, ninp) for i in DEMOVARS['LOCATION']])
        if use_demo and useone in ['religion', None]:
            self.religion_embed = torch.nn.ModuleList([nn.Embedding(ntoken, ninp) for i in DEMOVARS['RELIGION']])
        if use_demo and useone in ['gender', None]:
            self.gender_embed = torch.nn.ModuleList([nn.Embedding(ntoken, ninp) for i in DEMOVARS['GENDER']])

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
            # self.age_decode.weight = self.age_embed.weight
            # self.location_decode.weight = self.location_embed.weight
            # self.religion_decode.weight = self.religion_embed.weight
            # self.gender_decode.weight = self.gender_embed.weight

        self.use_pre = use_pre
        self.use_demo = use_demo
        self.useone = useone
        self.demouse = demouse
        self.init_weights(demoembs, mainmatrix)

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

    def init_weights(self, demoembs, mainmatrix):
        initrange = 0.1
        if not self.use_pre:
            self.cprint('Initializing encoder weights to random vectors...')
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.weight.data.uniform_(-initrange, initrange)

            if self.use_demo:
                if self.useone in ['age', None]:
                    for i in range(len(self.age_embed)):
                        self.age_embed[i].weight.data.uniform_(-initrange, initrange)
                if self.useone in ['location', None]:
                    for i in range(len(self.location_embed)):
                        self.location_embed[i].weight.data.uniform_(-initrange, initrange)
                if self.useone in ['religion', None]:
                    for i in range(len(self.religion_embed)):
                        self.religion_embed[i].weight.data.uniform_(-initrange, initrange)
                if self.useone in ['gender', None]:
                    for i in range(len(self.gender_embed)):
                        self.gender_embed[i].weight.data.uniform_(-initrange, initrange)

            # self.age_decode.weight.data.uniform_(-initrange, initrange)
            # self.location_decode.weight.data.uniform_(-initrange, initrange)
            # self.religion_decode.weight.data.uniform_(-initrange, initrange)
            # self.gender_decode.weight.data.uniform_(-initrange, initrange)
        else:
            self.cprint('Initializing encoder weights to pretrained embeddings...')
            # Try changing mainmatrix and see if performance changes when the MAIN matrix changes
            self.encoder.weight.data = torch.FloatTensor(demoembs[mainmatrix][-1])

            # Initialize the demographic embedding matrices
            if self.use_demo:
                if self.useone in ['age', None]:
                    self.cprint('Initializing age matrices...')
                    for ik in range(len(DEMOVARS['AGE'])):
                        self.age_embed[ik].weight.data = torch.FloatTensor(demoembs[0][ik])

                if self.useone in ['location', None]:
                    self.cprint('Initializing location matrices...')
                    for ik in range(len(DEMOVARS['LOCATION'])):
                        self.location_embed[ik].weight.data = torch.FloatTensor(demoembs[1][ik])

                if self.useone in ['religion', None]:
                    self.cprint('Initializing religion matrices...')
                    for ik in range(len(DEMOVARS['RELIGION'])):
                        self.religion_embed[ik].weight.data = torch.FloatTensor(demoembs[2][ik])

                if self.useone in ['gender', None]:
                    self.cprint('Initializing gender matrices...')
                    for ik in range(len(DEMOVARS['GENDER'])):
                        self.gender_embed[ik].weight.data = torch.FloatTensor(demoembs[3][ik])

        self.decoder.bias.data.fill_(0)
        # self.age_decode.bias.data.fill_(0)
        # self.location_decode.bias.data.fill_(0)
        # self.religion_decode.bias.data.fill_(0)
        # self.gender_decode.bias.data.fill_(0)

    def forward(self, input, hidden, ages, locations, religions, genders, return_h=False):
        emb, mask = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = emb.cuda()
        emb = self.lockdrop(emb, self.dropouti)

        if self.use_demo and self.useone == None:
            aemb = []
            lemb = []
            remb = []
            gemb = []
            for ik in range(len(ages)):
                # print('ik: '+ str(ik))
                # print('ages: '+ str(ages))
                # print('ages[ik]: '+ str(ages[ik]))
                # print('input[ik]: '+ str(input[ik]))
                taemb = []
                tlemb = []
                tremb = []
                tgemb = []
                for qk in range(len(ages[ik])):
                    aex, _ = embedded_dropout(self.age_embed[ages[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    lex, _ = embedded_dropout(self.location_embed[locations[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    rex, _ = embedded_dropout(self.religion_embed[religions[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    gex, _ = embedded_dropout(self.gender_embed[genders[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    taemb.append(aex)
                    tlemb.append(lex)
                    tremb.append(rex)
                    tgemb.append(gex)
                aemb.append(torch.stack(taemb))
                lemb.append(torch.stack(tlemb))
                remb.append(torch.stack(tremb))
                gemb.append(torch.stack(tgemb))

            aemb = torch.stack(aemb).cuda()
            lemb = torch.stack(lemb).cuda()
            remb = torch.stack(remb).cuda()
            gemb = torch.stack(gemb).cuda()

            # print('aemb.shape: ' + str(aemb.shape))

            if self.demouse == 'cat':
                raw_output = torch.cat([emb, aemb, lemb, remb, gemb], 2)
            elif self.demouse == 'sum':
                raw_output = emb + aemb + lemb + remb + gemb
        elif self.use_demo and self.useone != None:
            xemb = []
            for ik in range(len(ages)):
                txemb = []
                for qk in range(len(ages[ik])):
                    if self.useone == 'age':
                        xex, _ = embedded_dropout(self.age_embed[ages[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    elif self.useone == 'location':
                        xex, _ = embedded_dropout(self.location_embed[locations[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    elif self.useone == 'religion':
                        xex, _ = embedded_dropout(self.religion_embed[religions[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)
                    elif self.useone == 'gender':
                        xex, _ = embedded_dropout(self.gender_embed[genders[ik][qk]], input[ik][qk], dropout=self.dropoute if self.training else 0, mask=mask)

                    txemb.append(xex)
                xemb.append(torch.stack(txemb))
            xemb = torch.stack(xemb)
            if self.demouse == 'cat':
                raw_output = torch.cat([emb, xemb], 2)
            elif self.demouse == 'sum':
                raw_output = emb + xemb

        else:
            aemb = None
            lemb = None
            remb = None
            gemb = None
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
        # aemb = aemb.view(aemb.size(0)*aemb.size(1), aemb.size(2))
        # lemb = lemb.view(lemb.size(0)*lemb.size(1), lemb.size(2))
        # remb = remb.view(remb.size(0)*remb.size(1), remb.size(2))
        # gemb = gemb.view(gemb.size(0)*gemb.size(1), gemb.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs#, aemb, lemb, remb, gemb
        return result, hidden#, aemb, lemb, remb, gemb

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
