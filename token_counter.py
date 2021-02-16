

import json, os

from tqdm import tqdm
from collections import defaultdict

def main():
    top_speakers = []
    with open('top_speakers') as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)

    count_map = defaultdict(lambda: 0)
    for file in top_speakers:
        print('Opening file ' + file + '...')
        with open('all_posts/' + file + '_sts') as handle:
            lines = handle.readlines()
            for line in tqdm(lines):
                tline = line.strip().split(' ')
                for token in tline:
                    count_map[token] += 1

    with open('token_counts', 'w') as out_file:
        out_file.write(json.dumps(count_map))

if __name__ == '__main__':
    main()
