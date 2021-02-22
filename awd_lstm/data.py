
import os
import torch
import numpy as np

from tqdm import tqdm
from collections import Counter

from utils import AGES, LOCATIONS, RELIGIONS, GENDERS, DEMOVARS


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

# Pretrained embs path should be from ../data/demographic.embeddings.jshuf.100d.SUFFIX
# Assumes that each embedding file contains all words in the vocabulary and that they occur before the attribute-specific instances
class Corpus(object):
    def __init__(self, path, embs, emsize, cprint, test_file, pretrained=False, novf=False, from_gensim=False):
        self.dictionary = Dictionary()
        self.novf = novf
        cprint('Filter the vocabulary based on embedding file: ' + str(not self.novf))

        # The +1 last entry is the MAIN entry
        self.a_embeds = [{} for i in range(len(DEMOVARS['AGE'])+1)]
        self.l_embeds = [{} for i in range(len(DEMOVARS['LOCATION'])+1)]
        self.r_embeds = [{} for i in range(len(DEMOVARS['RELIGION'])+1)]
        self.g_embeds = [{} for i in range(len(DEMOVARS['GENDER'])+1)]
        demo_embed_map = {'AGE': self.a_embeds, 'LOCATION': self.l_embeds, 'RELIGION': self.r_embeds, 'GENDER': self.g_embeds}

        self.prevocab = {}
        for emb_typeU in demo_embed_map.keys():
            emb_type = emb_typeU.lower()
            cprint('Loading pretrained file: ' + embs + emb_type)

            with open(embs + emb_type) as handle:
                lines = handle.readlines()
                for line in lines[1:]:
                    tline = line.split(' ')

                    if len(tline) != emsize+2 and pretrained:
                        continue # should only be the first line
                    else:
                        dtype = tline[0]
                        word = tline[1]
                        if line.startswith('MAIN'):
                            self.prevocab[word] = 1
                        demo_map_idx = DEMOVARS[emb_typeU].index(dtype) if dtype in DEMOVARS[emb_typeU] else len(DEMOVARS[emb_typeU])
                        temp_demo_val = np.array(tline[2:]).astype(np.float) if pretrained else 1
                        demo_embed_map[emb_typeU][demo_map_idx][word] = temp_demo_val

        cprint('Pretrained vocab size is: ' + str(len(self.prevocab)))

        cprint('Tokenizing training...')
        self.train = self.tokenize(os.path.join(path, 'train.txt'), cprint)
        cprint('Tokenizing validation...')
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), cprint)
        cprint('Tokenizing test...')
        self.test = self.tokenize(os.path.join(path, 'test.txt'), cprint)
        if test_file != None:
            cprint('Tokenizing new test...')
            self.test2 = self.tokenize(os.path.join(path, test_file + '.txt'), cprint)

        # After constructing train/valid/test convert vectors to torch embedding matrix
        if pretrained:
            for emb_typeU in demo_embed_map.keys():
                emb_type = emb_typeU.lower()
                for ik in range(len(demo_embed_map[emb_typeU])):
                    key = 'MAIN' if ik == len(DEMOVARS[emb_typeU]) else DEMOVARS[emb_typeU][ik]
                    cprint('Constructing pretrained embedding matrix for type: ' + emb_type + ' and ' + key)
                    temp_embs = np.zeros((len(self.dictionary), emsize))
                    for i in range(len(self.dictionary)):
                        if self.dictionary.idx2word[i] in self.prevocab:
                            temp_embs[i] = demo_embed_map[emb_typeU][ik][self.dictionary.idx2word[i]]
                        else:
                            cprint('Generating a random intial embedding for \'' + str(self.dictionary.idx2word[i]) + '\'')
                            temp_embs[i] = np.random.normal(scale=0.6, size=(emsize, ))
                    
                    # Replace the embedding dictionary with the temp matrix
                    demo_embed_map[emb_typeU][ik] = temp_embs
        else:
            cprint('Setting demographic matricies to zero...')
            self.a_embeds = np.zeros((len(DEMOVARS['AGE']), len(self.dictionary), emsize))
            self.g_embeds = np.zeros((len(DEMOVARS['GENDER']), len(self.dictionary), emsize))
            self.l_embeds = np.zeros((len(DEMOVARS['LOCATION']), len(self.dictionary), emsize))
            self.r_embeds = np.zeros((len(DEMOVARS['RELIGION']), len(self.dictionary), emsize))

        # Free memory
        del self.prevocab

    def tokenize(self, path, cprint):
        assert os.path.exists(path)
        # Cache file lines
        tlines = []
        tages = []
        tlocs = []
        trels = []
        tgens = []

        tokens = 0
        # Add words to the dictionary
        with open(path, 'r') as f:
            lcount = 0
            lines = f.readlines()
            for line in tqdm(lines):
                tline = line.split('\t')
                # Data has format tab separated: 0, age, country, religion, gender, text
                if len(tline) != 6:
                    cprint('Error reading line ' + str(lcount) + ' with ' + str(len(tline)) + 'parts, skipping...')
                    continue

                tages.append(DEMOVARS['AGE'].index(tline[1]))
                trels.append(DEMOVARS['RELIGION'].index(tline[3]))
                tgens.append(DEMOVARS['GENDER'].index(tline[4]))

                te_index = DEMOVARS['LOCATION'].index(tline[2]) if tline[2] in DEMOVARS['LOCATION'] else DEMOVARS['LOCATION'].index('unknown')
                tlocs.append(te_index)

                words = [wi if wi in self.prevocab or self.novf else '<unk>' for wi in tline[5].split(' ') if wi.strip() != ''] + ['<eos>']
                tlines.append(words)
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                lcount += 1

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        aids = torch.LongTensor(tokens)
        lids = torch.LongTensor(tokens)
        rids = torch.LongTensor(tokens)
        gids = torch.LongTensor(tokens)

        token = 0
        lcount = 0
        cura = -1
        curl = -1
        curr = -1
        curg = -1
        for words in tlines:
            cura = tages[lcount]
            curl = tlocs[lcount]
            curr = trels[lcount]
            curg = tgens[lcount]
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                aids[token] = cura
                lids[token] = curl
                rids[token] = curr
                gids[token] = curg
                token += 1
            lcount += 1

        return ids, aids, lids, rids, gids
