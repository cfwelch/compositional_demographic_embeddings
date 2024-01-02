

import operator, datetime, base36, json, sys, os, re

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

from utils import text_clean, DIR_SET

# This originally merged files for plaintext posts but now is changed to use JSON
def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--speakers', dest='speakers', help='Name of the file containing the speakers.', default='top_speakers', type=str)
    opt = parser.parse_args()

    os.makedirs('all_posts', exist_ok=True)

    top_speakers = []
    with open(opt.speakers) as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)
    top_speakers = set(top_speakers)

    all_posts = {ts: 0 for ts in top_speakers}
    for cur_dir in DIR_SET:
        # Reset for each directory so you can append instead of storing all in memory
        utt_sets = {ts: [] for ts in top_speakers}

        file_list = [i for i in os.listdir('data/' + cur_dir) if i.endswith('tposts')]
        print('File List: ' + str(file_list))

        for i in file_list:
            post_map = defaultdict(lambda: [])

            print('Opening data/' + cur_dir + '/' + i + '...')
            with open('data/' + cur_dir + '/' + i) as handle:
                fposts = json.loads(handle.read())

                for speaker in fposts:
                    utt_sets[speaker].extend(fposts[speaker])

        # Used to do this after reading all the files but required too much memory for complete_authors
        print('Writing speakers to files...')
        for speaker in utt_sets:
            all_posts[speaker] += len(utt_sets[speaker])
            with open('all_posts/' + speaker + '_json', 'a') as handle:
                for line in utt_sets[speaker]:
                    handle.write(json.dumps(line) + '\n')

    with open('demographic/ppl_all', 'w') as handle:
        for ts in all_posts:
            handle.write(ts + '\t' + str(all_posts[ts]) + '\n')

if __name__ == '__main__':
    main()
