

import datetime, socket, torch, time, html, re, os

from termcolor import colored

dir_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(dir_path + '/logs', exist_ok=True)

AGES = ['young', 'old', 'unknown']
LOCATIONS = ['usa', 'asia', 'oceania', 'uk', 'europe', 'canada', 'unknown']
RELIGIONS = ['atheist', 'buddhist', 'christian', 'hindu', 'muslim', 'unknown']
GENDERS = ['male', 'female', 'unknown']
DEMOVARS = {'AGE': AGES, 'LOCATION': LOCATIONS, 'RELIGION': RELIGIONS, 'GENDER': GENDERS}

def repackage_hidden(h):
    # Wraps hidden states in new Tensors, to detach them from their history.
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(inputs, bsz, args):
    w, a, l, r, g = inputs

    def dobatchify(data):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if args.cuda:
            data = data.cuda()
        return data

    w = dobatchify(w)
    a = dobatchify(a)
    l = dobatchify(l)
    r = dobatchify(r)
    g = dobatchify(g)

    return w, a, l, r, g

def get_batch(inputs, i, args, seq_len=None, evaluation=False):
    source, a, l, r, g = inputs
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    ta = a[i:i+seq_len]
    tl = l[i:i+seq_len]
    tr = r[i:i+seq_len]
    tg = g[i:i+seq_len]
    return data, target, ta, tl, tr, tg

def gprint(msg, logname, error=False, important=False, ptime=True, p2c=True):
    tmsg = msg if not important else colored(msg, 'cyan')
    tmsg = tmsg if not error else colored(msg, 'red')
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    cmsg = str(st) + ': ' + str(tmsg) if ptime else str(tmsg)
    tmsg = str(st) + ': ' + str(msg) if ptime else str(msg)
    if p2c:
        print(cmsg)
    log_file = open(dir_path + '/logs/' + logname + '.log', 'a')
    log_file.write(tmsg + '\n')
    log_file.flush()
    log_file.close()
