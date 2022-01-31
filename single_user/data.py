
import os
import torch
import numpy as np

from tqdm import tqdm
from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path, uname, embs, emsize, cprint, pretrained=False, novf=False, from_gensim=False, ts_list=None, second_auth=None):
        self.dictionary = Dictionary()
        self.embeddings = None
        self.novf = novf
        cprint('Filter the vocabulary based on embedding file: ' + str(not self.novf))

        self.u_embeds = {}

        self.prevocab = {}

        with open(embs + 'MAIN_ctx_embeds') as handle:
            lines = handle.readlines()
            for line in lines[1:]:
                tline = line.split(' ')

                if len(tline) != emsize+1:
                    continue # should only be the first line
                else:
                    word = tline[0]
                    self.prevocab[word] = np.array(tline[1:]).astype(np.float) if pretrained else 1

        with open(embs + uname + '_ctx_embeds') as handle:
            lines = handle.readlines()
            for line in lines[1:]:
                tline = line.split(' ')

                if len(tline) != emsize+1:
                    continue # should only be the first line
                else:
                    word = tline[0]
                    self.u_embeds[word] = np.array(tline[1:]).astype(np.float) if pretrained else 1

        cprint('Pretrained vocab size is: ' + str(len(self.prevocab)))

        cprint('Tokenizing training...')
        self.train = self.tokenize(os.path.join(path + uname, 'train.txt'), cprint)
        cprint('Tokenizing validation...')
        self.valid = self.tokenize(os.path.join(path + uname, 'valid.txt'), cprint)
        cprint('Tokenizing test...')
        self.test = self.tokenize(os.path.join(path + uname, 'test.txt'), cprint)

        if second_auth != None:
            # cprint('2nd Author Tokenizing training...')
            # self.train2 = self.tokenize(os.path.join(path + second_auth, 'train.txt'), cprint)
            cprint('2nd Author Tokenizing validation...')
            self.valid2 = self.tokenize(os.path.join(path + second_auth, 'valid.txt'), cprint, limit_voc=True)
            # cprint('2nd Author Tokenizing test...')
            # self.test2 = self.tokenize(os.path.join(path + second_auth, 'test.txt'), cprint)

        # Tokenize all top author files for authorship attribution
        self.ts = {}
        self.ts_lines = {}
        if ts_list != None:
            for i in range(len(ts_list)):
                aa_path = os.path.join(path + ts_list[i] + '/', 'aa.txt')
                self.ts[i] = self.tokenize(aa_path, cprint, limit_voc=True)
                with open(aa_path) as handle:
                    self.ts_lines[i] = handle.readlines()
    
        # After constructing train/valid/test convert vectors to torch embedding matrix
        if pretrained:
            cprint('Constructing pretrained embedding matrix...')
            self.embeddings = np.zeros((len(self.dictionary), emsize))
            for i in range(len(self.dictionary)):
                if self.dictionary.idx2word[i] in self.prevocab:
                    self.embeddings[i] = self.prevocab[self.dictionary.idx2word[i]]
                else:
                    cprint('Generating a random intial embedding for \'' + str(self.dictionary.idx2word[i]) + '\'')
                    self.embeddings[i] = np.random.normal(scale=0.6, size=(emsize, ))

            temp_embs = np.zeros((len(self.dictionary), emsize))
            for i in range(len(self.dictionary)):
                if self.dictionary.idx2word[i] in self.prevocab:
                    temp_embs[i] = self.u_embeds[self.dictionary.idx2word[i]]
                else:
                    cprint('Generating a random intial embedding for \'' + str(self.dictionary.idx2word[i]) + '\'')
                    temp_embs[i] = np.random.normal(scale=0.6, size=(emsize, ))
            self.u_embeds = temp_embs

        # Free memory
        del self.prevocab

    def tokenize(self, path, cprint, limit_voc=False):
        assert os.path.exists(path)
        # Cache file lines
        tlines = []
        tusers = []

        tokens = 0
        # Add words to the dictionary
        with open(path, 'r') as f:
            lcount = 0
            lines = f.readlines()
            for line in tqdm(lines):
                tline = line.strip()
                tusers.append(0)
                words = [wi if wi in self.prevocab or self.novf else '<unk>' for wi in tline.split(' ')] + ['<eos>']
                if limit_voc:
                    words = [wi if wi in self.dictionary.word2idx else '<unk>' for wi in words]
                tlines.append(words)
                tokens += len(words)

                if not limit_voc:
                    for word in words:
                        self.dictionary.add_word(word)
                lcount += 1

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        uids = torch.LongTensor(tokens)

        token = 0
        lcount = 0
        curu = -1
        for words in tlines:
            curu = tusers[lcount]
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                uids[token] = curu
                token += 1
            lcount += 1

        return ids, uids
