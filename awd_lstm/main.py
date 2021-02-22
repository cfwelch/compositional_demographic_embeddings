
import argparse, operator, time, math, sys
import numpy as np
import torch
import torch.nn as nn

import data, model

from collections import defaultdict
from utils import batchify, get_batch, repackage_hidden, gprint, AGES, LOCATIONS, RELIGIONS, GENDERS

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

# Added parameters
parser.add_argument('--usepre', default=False, help='Use pretrained embeddings for words and demographic or just vocab', action='store_true')
parser.add_argument('--usedemo', default=False, help='Use demographic information in prediction', action='store_true')
parser.add_argument('--tied', default=False, help='Tie input and output weights for the word embeddings', action='store_true')
parser.add_argument('--burnin', type=int, default=0, help='Number of epochs to freeze embeddings (should only be used for pretrained embeddings)')
parser.add_argument('--pre', type=str, default=None, help='Location of the pretrained embeddings to load')
parser.add_argument('--novf', default=False, help='If set, do not filter the vocabulary to only contain words in --pre', action='store_true')
parser.add_argument('--gensim', default=False, help='If set, assums first token is word and rest are dimensions of embedding', action='store_true')
parser.add_argument('--demouse', type=str, default='sum', help='Type of operation to do with demographic embeddings, can be \'sum\' or \'cat\'')
parser.add_argument('--useone', type=str, default=None, help='If not None uses only one demographic attribute instead of all four: [age, location, religion, gender]')
parser.add_argument('--mainmatrix', type=int, default=0, help='Matrix index for MAIN embeddings to use, 0=age, 1=location, 2=religion, 3=gender')
parser.add_argument('--infreeze', default=False, help='Do you want to also freeze the encoder?', action='store_true')
parser.add_argument('--test', type=str, default=None, help='Use if you have a test file you want to use other than \'test\'')
# parser.add_argument('--match_input_size', default=False, help='Copy the input to the first LSTM layer to match size of concatenated embeddings (*5)', action='store_true')
args = parser.parse_args()
# args.tied = True


FREEZE_SET = ['age_embed', 'location_embed', 'religion_embed', 'gender_embed']
if args.infreeze:
    FREEZE_SET.append('encoder')
if args.tied:
    FREEZE_SET.append('decoder')

def cprint(msg, error=False, important=False):
    log_name = 'log_' + ('base' if not args.usepre else 'pre') + '_useone-' + str(args.useone) + '_' + args.save
    gprint(msg, log_name, error, important)

def wprint(msg, error=False, important=False):
    log_name = 'ppl_diffs_' + ('base' if not args.usepre else 'pre') + '_useone-' + str(args.useone) + '_' + args.save
    gprint(msg, log_name, error, important, ptime=False, p2c=False)

def aprint(msg, error=False, important=False):
    log_name = 'all_ppls_' + ('base' if not args.usepre else 'pre') + '_useone-' + str(args.useone) + '_' + args.save
    gprint(msg, log_name, error, important, ptime=False, p2c=False)

if args.demouse not in ['cat', 'sum']:
    cprint('Error: demouse must be \'cat\' or \'sum\'...')
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
if os.path.exists(fn):
    cprint('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    cprint('Producing dataset...')
    corpus = data.Corpus(args.data, args.pre, args.emsize, cprint, args.test, args.usepre, args.novf, from_gensim=args.gensim)
    # torch.save(corpus, fn) # I think forgetting I cached this has caused me more problems than it's worth

eval_batch_size = 10
test_batch_size = 1
cprint('Train length: ' + str(len(corpus.train)))
cprint('Valid length: ' + str(len(corpus.valid)))
cprint('Test length: ' + str(len(corpus.test)))
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
if args.test != None:
    test2_data = batchify(corpus.test2, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args.usepre, args.usedemo, args.useone, (corpus.a_embeds, corpus.l_embeds, corpus.r_embeds, corpus.g_embeds), args.demouse, args.mainmatrix, cprint)#, args.match_input_size
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
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    cprint('Using splits ' + str(criterion.splits))
###
if args.cuda:
    # Because we have more embedding matrices we have to load them on CPU
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

def evaluate(data_source, batch_size=10, return_breakdown=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    loss_types = {k: defaultdict(lambda: []) for k in ['age', 'location', 'religion', 'gender']}
    word_ppls = defaultdict(lambda: [])
    word_ppls_by_type = {k: defaultdict(lambda: defaultdict(lambda: [])) for k in ['age', 'location', 'religion', 'gender']}

    for i in range(0, data_source[0].size(0) - 1, args.bptt):
        data, targets, ages, locations, religions, genders = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden, ages, locations, religions, genders)
        tlout, llout = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        total_loss += len(data) * tlout.data
        hidden = repackage_hidden(hidden)

        # Handle breakdown of losses
        if len(llout) > 1:
            cprint('length of llout: ' + str(len(llout)))
        tllz = llout[0].squeeze()
        ages = ages.view(-1)
        locations = locations.view(-1)
        religions = religions.view(-1)
        genders = genders.view(-1)
        data = data.view(-1)
        targets = targets.view(-1)

        assert len(data) == len(genders)
        for zz in range(len(data)):
            # print(str(ages[zz].item()) + ' -- ' + str(tllz[zz].item()) + ' -- ' + str(corpus.dictionary.idx2word[data[zz].item()]))
            loss_types['age'][ages[zz].item()].append(tllz[zz].item())
            loss_types['location'][locations[zz].item()].append(tllz[zz].item())
            loss_types['religion'][religions[zz].item()].append(tllz[zz].item())
            loss_types['gender'][genders[zz].item()].append(tllz[zz].item())

            if return_breakdown:
                the_word = corpus.dictionary.idx2word[targets[zz].item()]
                word_ppls[the_word].append(tllz[zz].item())
                word_ppls_by_type['age'][ages[zz].item()][the_word].append(tllz[zz].item())
                word_ppls_by_type['location'][locations[zz].item()][the_word].append(tllz[zz].item())
                word_ppls_by_type['religion'][religions[zz].item()][the_word].append(tllz[zz].item())
                word_ppls_by_type['gender'][genders[zz].item()][the_word].append(tllz[zz].item())

    if return_breakdown:
        return total_loss.item() / len(data_source[0]), loss_types, word_ppls, word_ppls_by_type
    else:
        return total_loss.item() / len(data_source[0]), loss_types

def print_word_ppl_by_dem(wpbt, t, lk):
    for key in wpbt[t]:
        wprint('==' + t + '-' + lk[key] + '==')
        for wk in wpbt[t][key]:
            wpbt[t][key][wk] = math.exp(sum(wpbt[t][key][wk]) / len(wpbt[t][key][wk]))
        for k,v in sorted(wpbt[t][key].items(), key=operator.itemgetter(1), reverse=True):
            wprint(k + '\t' + str(v))

def print_losses(losses, name):
    for key in losses['age']:
        cprint('| ' + name + ' ppl for age ' + str(AGES[key]) + ': ' + str(math.exp(sum(losses['age'][key]) / len(losses['age'][key]))))
    for key in losses['location']:
        cprint('| ' + name + ' ppl for location ' + str(LOCATIONS[key]) + ': ' + str(math.exp(sum(losses['location'][key]) / len(losses['location'][key]))))
    for key in losses['religion']:
        cprint('| ' + name + ' ppl for religion ' + str(RELIGIONS[key]) + ': ' + str(math.exp(sum(losses['religion'][key]) / len(losses['religion'][key]))))
    for key in losses['gender']:
        cprint('| ' + name + ' ppl for gender ' + str(GENDERS[key]) + ': ' + str(math.exp(sum(losses['gender'][key]) / len(losses['gender'][key]))))

def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    # for i in hidden:
    #     print('i is a ' + str(type(i)))
    #     for ii in i:
    #         print('ii is a ' + str(type(ii)))
    #         ii = ii.cuda()
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
        data, targets, ages, locations, religions, genders = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, ages, locations, religions, genders, return_h=True)
        raw_loss, train_loss_list = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        if len(train_loss_list) > 1:
            cprint('length of train_ll: ' + str(len(train_loss_list)))

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
            # print('encode weight: ' + str(model.encoder.weight.data))
            # print('decode weight: ' + str(model.decoder.weight.data))
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

            val_loss2, losses = evaluate(val_data)
            cprint('-' * 89)
            cprint('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print_losses(losses, 'valid')
            cprint('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                cprint('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                if prm in tmp:
                    prm.data = tmp[prm].clone()

        else:
            val_loss, losses = evaluate(val_data, eval_batch_size)
            cprint('-' * 89)
            cprint('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print_losses(losses, 'valid')
            cprint('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                cprint('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                cprint('Switching to ASGD')
                optimizer = torch.optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                cprint('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                cprint('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    cprint('-' * 89)
    cprint('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss, losses, word_ppls, word_ppls_by_type = evaluate(test_data, test_batch_size, return_breakdown=True)
cprint('=' * 89)
cprint('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(test_loss, math.exp(test_loss), test_loss / math.log(2)))
print_losses(losses, 'test')
for key in word_ppls:
    for key2 in word_ppls[key]:
        aprint(key + '\t' + str(key2))
    word_ppls[key] = math.exp(sum(word_ppls[key]) / len(word_ppls[key]))

wprint('==overall==')
for k,v in sorted(word_ppls.items(), key=operator.itemgetter(1), reverse=True):
    wprint(k + '\t' + str(v))

print_word_ppl_by_dem(word_ppls_by_type, 'age', AGES)
print_word_ppl_by_dem(word_ppls_by_type, 'location', LOCATIONS)
print_word_ppl_by_dem(word_ppls_by_type, 'religion', RELIGIONS)
print_word_ppl_by_dem(word_ppls_by_type, 'gender', GENDERS)
cprint('=' * 89)

if args.test != None:
    test_loss2, losses = evaluate(test2_data, test_batch_size)
    cprint('=' * 89)
    cprint('| End of training | test2 loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(test_loss2, math.exp(test_loss2), test_loss2 / math.log(2)))
    print_losses(losses, 'test2')
    cprint('=' * 89)
