

import unicodedata, operator, datetime, base36, html, json, sys, os
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from utils import replace_urls

# This file will remove posts by the top speakers which are in the 'counting' subreddit but will also unescape the HTML because that is too hard to do in Python 2 when we need the sentences tokenized with the Stanford CoreNLP wrapper.
def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--speakers', dest='speakers', help='Name of the file containing the speakers.', default='top_speakers', type=str)
    parser.add_argument('-l', '--linear_post_subset', dest='linear_post_subset', help='Use linear post subset instead of all_posts.', default=False, action='store_true')
    opt = parser.parse_args()

    top_speakers = []
    with open(opt.speakers) as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)

    file_set = ['all_posts/' + user + '_json' for user in top_speakers]

    rposts = False
    if rposts:
        # rposts_sample is not json formatted?
        file_set.append('all_posts/rposts_json')

    if opt.linear_post_subset:
        file_set = ['linear_post_subset']

    print('Reading post set...')
    for filename in file_set:
        out_json = open(filename + '_filtered', 'w')
        print('Filtering file ' + filename + '...')
        with open(filename) as handle:
            lines = handle.readlines()
            for line in tqdm(lines):
                tline = json.loads(line)
                tsubreddit = tline['subreddit']
                tline_body = html.unescape(tline['body'])
                tline_body = replace_urls(tline_body)
                tline_body = ''.join(c for c in unicodedata.normalize('NFD', tline_body) if unicodedata.category(c) != 'Mn')
                tauthor = str(tline['author'])
                tline['body'] = tline_body
                distinguished = tline['distinguished'] if 'distinguished' in tline else None

                if opt.linear_post_subset:
                    if tsubreddit != 'counting':
                        out_json.write(json.dumps(tline) + '\n')
                else:
                    if (tauthor in top_speakers or rposts) and distinguished == None and tsubreddit != 'counting':
                        out_json.write(json.dumps(tline) + '\n')
        out_json.close()

if __name__ == '__main__':
    main()
