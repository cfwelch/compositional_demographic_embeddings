import argparse
import time, math, sys
import numpy as np
import torch
import torch.nn as nn

import data
import model

from utils import batchify, get_batch, repackage_hidden, gprint, dir_path, TOP_SPEAKERS

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=70, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--nonmono', type=int, default=5, help='random seed')
parser.add_argument('--cuda', default=True, action='store_false', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt', help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='', help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd', help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

# new param
parser.add_argument('--usepre', default=False, help='Use pretrained embeddings for words and individual or just vocab', action='store_true')
parser.add_argument('--useind', default=False, help='Use individual information in prediction', action='store_true')
parser.add_argument('--tied', default=False, help='Tie input and output weights for the word embeddings', action='store_true')
parser.add_argument('--burnin', type=int, default=0, help='Number of epochs to freeze embeddings (should only be used for pretrained embeddings)')
parser.add_argument('--pre', type=str, default=None, help='Location of the pretrained embeddings to load')
parser.add_argument('--novf', default=False, help='If set, do not filter the vocabulary to only contain words in --pre', action='store_true')
parser.add_argument('--gensim', default=False, help='If set, assums first token is word and rest are dimensions of embedding', action='store_true')
parser.add_argument('--induse', type=str, default='sum', help='Type of operation to do with individual embeddings, can be \'sum\' or \'cat\'')
parser.add_argument('--infreeze', default=False, help='Do you want to also freeze the encoder?', action='store_true')
parser.add_argument('--name', default='', type=str, help='Name of user to use')
parser.add_argument('--aaeval', default=False, help='Evaluate on all aa.txt files', action='store_true')
args = parser.parse_args()
# args.tied = True

FREEZE_SET = ['user_embed']
if args.infreeze:
    FREEZE_SET.append('encoder')
if args.tied:
    FREEZE_SET.append('decoder')

def cprint(msg, error=False, important=False):
    log_name = 'log_' + ('base' if not args.usepre else 'pre') + '_' + args.name + '.pt'
    gprint(msg, log_name, error, important)

if args.induse not in ['cat', 'sum']:
    cprint('Error: induse must be \'cat\' or \'sum\'...')
    sys.exit(0)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        cprint("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = ('corpus.{}.data' + ('.pre' if args.usepre else '.new')).format(hashlib.md5(args.data.encode()).hexdigest())
cprint('Producing dataset...')

ts_list = None
if args.aaeval:
    ts_list = [i for i in TOP_SPEAKERS if i != '-rix']
corpus = data.Corpus(args.data, args.name, args.pre, args.emsize, cprint, args.usepre, args.novf, from_gensim=args.gensim, ts_list=ts_list)

eval_batch_size = 10
test_batch_size = 1
cprint('Train length: ' + str(len(corpus.train)))
cprint('Valid length: ' + str(len(corpus.valid)))
cprint('Test length: ' + str(len(corpus.test)))
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args.usepre, args.useind, (corpus.embeddings, corpus.u_embeds), args.induse, cprint)
###
if args.resume:
    cprint('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    criterion = SplitCrossEntropyLoss(args.nhid if not args.tied else args.emsize, splits=splits, verbose=False)
    cprint('Using splits ' + str(criterion.splits))
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
cprint('Args: ' + str(args))
cprint('Model total parameters: ' + str(total_params))

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ent_list = []
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source[0].size(0) - 1, args.bptt):
        data, targets, users = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden, users)
        oloss, ent = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        ent_list.extend(ent.cpu().tolist())
        total_loss += len(data) * oloss.data
        hidden = repackage_hidden(hidden)

    ent_list = [i[0] for i in ent_list]
    return total_loss.item() / len(data_source[0]), ent_list


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data[0].size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets, users = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, users, return_h=True)
        raw_loss, entt = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            cprint('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(epoch, batch, len(train_data[0]) // args.bptt, optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000



# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.burnin <= 0:
        FREEZE_SET = []

    # Set the frozen embedding layers
    for name, child in model.named_children():
        if name in FREEZE_SET:
            cprint(name + '\tis frozen')
            for param in child.parameters():
                param.requires_grad = False
        else:
            cprint(name + '\tis unfrozen')
            for param in child.parameters():
                param.requires_grad = True

    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):

        if epoch == args.burnin:
            # Unfreeze the embedding layers
            for name, child in model.named_children():
                if name in FREEZE_SET:
                    cprint(name + '\tis unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True

            # Reset the optimizer
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.wdecay)
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.wdecay)

        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2, ent2 = evaluate(val_data)
            cprint('-' * 89)
            cprint('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            cprint('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save + args.name + '.pt')
                cprint('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                if prm in tmp:
                    prm.data = tmp[prm].clone()

        else:
            val_loss, ent = evaluate(val_data, eval_batch_size)
            cprint('-' * 89)
            cprint('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            cprint('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save + args.name + '.pt')
                cprint('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                cprint('Switching to ASGD')
                optimizer = torch.optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                cprint('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save + args.name + '.pt', epoch))
                cprint('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    cprint('-' * 89)
    cprint('Exiting from training early')

# Load the best saved model.
model_load(args.save + args.name + '.pt')

# Run on test data or aaeval data
if args.aaeval:
    for i in range(len(ts_list)):
        tsi_set = batchify(corpus.ts[i], test_batch_size, args)
        tloss, ent = evaluate(tsi_set, test_batch_size)
        # cprint('ent type: ' + str(type(ent))) -- it's a list
        ent_sents = []
        cur_ent_sent = []
        counter = 0
        for j in range(len(corpus.ts_lines[i])):
            toks = corpus.ts_lines[i][j].strip().split(' ')
            for k in range(len(toks)+(0 if j == len(corpus.ts_lines[i])-1 else 1)):
                cur_ent_sent.append(ent[counter])
                counter += 1
            ent_sents.append(cur_ent_sent)
            cur_ent_sent = []

        with open(dir_path + '/logs_aa/' + args.name + '_on_' + ts_list[i], 'w') as handle:
            for ent_sent in ent_sents:
                es_avg = sum(ent_sent) / len(ent_sent)
                handle.write(str(es_avg) + ':' + str(ent_sent) + '\n')
else:
    test_loss, ent = evaluate(test_data, test_batch_size)
    cprint('=' * 89)
    cprint('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(test_loss, math.exp(test_loss), test_loss / math.log(2)))
    cprint('=' * 89)
