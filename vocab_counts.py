

import operator, json, os

from collections import defaultdict
from tqdm import tqdm

top_reddits = []
SPLIT_TYPE = 'user'
os.makedirs('vocabs/' + SPLIT_TYPE, exist_ok=True)

def main():
    top_speakers = []
    with open('top_speakers') as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)
    make_vocabs(top_speakers, top_reddits, SPLIT_TYPE)

def print_distributions(top_speakers):
    known_words = defaultdict(lambda: 'UNK')
    with open('pos_dist_out_above_95') as handle:
        for line in handle.readlines():
            tline = line.strip().split(' --- ')
            known_words[tline[0]] = tline[1]

    avg_pos_counts = defaultdict(lambda: 0)
    print('Iterating over files...')
    for file in top_speakers:
        pos_counts = defaultdict(lambda: 0)
        print('Opening file ' + file + '...')

        with open('all_posts/' + file + '_sts') as handle:
            lines = handle.readlines()
            for line in tqdm(lines):
                tline = line.strip().split(' ')
                for tok in tline:
                    pos_counts[known_words[tok]] += 1
                    avg_pos_counts[known_words[tok]] += 1

        total = sum(pos_counts.values())
        for k,v in sorted(pos_counts.items(), key=operator.itemgetter(1), reverse=True):
            print(k + '\t' + str(v) + '\t-\t' + '{:.2f}'.format(v*100.0/total) + '\t->\t' + '{:.2f}'.format(v*100.0/(total - pos_counts['UNK'])))

    print('\nAveraged POS counts: ')
    total = sum(avg_pos_counts.values())
    for k,v in sorted(avg_pos_counts.items(), key=operator.itemgetter(1), reverse=True):
        print(k + '\t' + str(v) + '\t-\t' + '{:.2f}'.format(v*100.0/total) + '\t->\t' + '{:.2f}'.format(v*100.0/(total - avg_pos_counts['UNK'])))

def make_vocabs(top_speakers, top_reddits, split_type):
    print('Iterating over files...')
    if split_type == 'user':
        vcounts = {k: defaultdict(lambda: 0) for k in top_speakers}
    elif split_type == 'subreddit':
        vcounts = {k: defaultdict(lambda: 0) for k in top_reddits}
        vcounts['Other'] = defaultdict(lambda: 0)

    for file in sorted(top_speakers):
        print('Opening file ' + file + '...')
        with open('all_posts/' + file + '_json_filtered_tokenized') as handle:
            lines = handle.readlines()
            for line in tqdm(lines):
                tline = json.loads(line)
                tkey = tline['author'] if split_type == 'user' else (tline['subreddit'] if tline['subreddit'] in top_reddits else 'Other')
                for tok in tline['body'].split(' '):
                    vcounts[tkey][tok] += 1

    for key in vcounts:
        with open('vocabs/' + split_type + '/' + key + '_vocab', 'w') as handle:
            for k,v in sorted(vcounts[key].items(), key=operator.itemgetter(1), reverse=True):
                handle.write(k + '\t' + str(v) + '\n')

if __name__ == '__main__':
    main()
