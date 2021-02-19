

import operator, datetime, base36, json, sys, os, re

from utils import get_post_id

from tqdm import tqdm
from random import random
from collections import defaultdict
from argparse import ArgumentParser

from utils import text_clean
from utils import DIR_SET as DIR_SET_ALL

DEFAULT_LOCATION = 'data/'

def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', help='The directory of files to map.', default=None, type=str)
    parser.add_argument('-p', '--posts', dest='posts', help='Get the post content instead of ids.', default=False, action='store_true')
    parser.add_argument('-r', '--random', dest='random', help='Get random post sample instead of top speaker posts.', default=False, action='store_true')
    parser.add_argument('-s', '--speakers', dest='speakers', help='Name of the file containing the speakers.', default='top_speakers', type=str)
    parser.add_argument('-rd', '--root_dir', dest='root_dir', help='Location of the post files.', default=DEFAULT_LOCATION, type=str)
    opt = parser.parse_args()

    SAMPLE_RATE = 0.01 / 17
    dir_set = [opt.dir] if opt.dir != 'all' else DIR_SET_ALL

    top_speakers = []
    with open(opt.speakers) as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)

    if opt.random:
        rfile = open('rposts', 'w')

    for cur_dir in dir_set:
        file_list = [i for i in os.listdir(opt.root_dir + cur_dir) if re.match('RC_\d\d\d\d-\d\d$', i)]
        print('File List: ' + str(file_list))

        for i in file_list:
            post_map = defaultdict(lambda: [])

            print('Opening ' + cur_dir + '/' + i + '...')
            with open(opt.root_dir + cur_dir + '/' + i) as handle:
                for line in tqdm(handle):
                    tline = json.loads(line)
                    tauthor = str(tline['author'])
                    post_id = get_post_id(tline['id'])

                    if tauthor in top_speakers and not opt.random:
                        #post_map[body_text] += 1
                        if opt.posts:
                            body_text = tline['body'] # text_clean
                            post_map[tauthor].append(tline)
                        else:
                            post_map[tauthor].append(post_id)
                    elif opt.random and random() < SAMPLE_RATE:
                        rfile.write(json.dumps(tline) + '\n')

            if not opt.random:
                suffix = '_tposts' if opt.posts else '_ts_pids'
                with open(opt.root_dir + cur_dir + '/' + i + suffix, 'w') as handle:
                    #for k,v in sorted(post_map.items(), key=operator.itemgetter(1), reverse=True):
                    if opt.posts:
                        handle.write(json.dumps(post_map))
                    else:
                        for k,v in post_map.items():
                            handle.write(str(k) + ':' + str(v) + '\n')

if __name__ == '__main__':
    main()
