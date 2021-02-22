

import operator, json, html, sys, os

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

VOCAB_FILE = 'reddit_user_vocab'

def main():
    top_speakers = {}
    with open('top_speakers') as handle:
        for line in handle:
            tline = line.strip().split('\t')
            top_speakers[tline[0]] = 0

    gen_prep_per_user(top_speakers)

def gen_prep_per_user(top_speakers):
    file_set = {'all_posts/' + user + '_sts': user for user in top_speakers}

    vocab = defaultdict(lambda: 0)
    out_file = open('sts_combined_users', 'w')
    print('Reading post set...')
    for filename, user in file_set.items():
        print('Combining file ' + filename + '...')
        with open(filename) as handle:
            lines = handle.readlines()
            for line in tqdm(lines):
                tline = line.strip().replace('\t', ' ')
                for word in tline.split(' '):
                    vocab[word] += 1
                out_file.write('0\t' + user + '\t' + tline + '\n')

    with open(VOCAB_FILE, 'w') as out_file:
        for k,v in sorted(vocab.items(), key=operator.itemgetter(1), reverse=True):
            out_file.write(str(v) + '\t' + k + '\n')

if __name__ == '__main__':
    main()
