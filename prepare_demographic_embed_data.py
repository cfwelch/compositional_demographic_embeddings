

import operator, json, os

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--speakers', dest='speakers', help='Name of the file containing the speakers.', default='demographic/complete_authors', type=str)
    parser.add_argument('-d', '--in_dir', dest='in_dir', help='Name of the directory with preprocessed json strings.', default='all_posts', type=str)
    parser.add_argument('-o', '--old_age', dest='old_age', help='Age to map integer to \'old\', else \'young\'.', default=30, type=int)
    parser.add_argument('-v', '--vocab', dest='vocab', help='Name for output vocabulary file.', default='reddit_vocab', type=str)
    opt = parser.parse_args()

    top_speakers = {}
    with open(opt.speakers) as handle:
        for line in handle:
            tline = line.strip().split('\t')
            name = tline[0]
            top_speakers[name] = {'age': tline[1] if tline[1] != 'UNK' else 'unknown', \
                                  'gender': tline[2] if tline[2] != 'UNK' else 'unknown', \
                                  'religion': tline[3] if tline[3] != 'UNK' else 'unknown', \
                                  'location': tline[4] if tline[4] != 'UNK' else 'unknown', \
                                  'posts': tline[5]}

            if top_speakers[name]['age'] != 'unknown':
                top_speakers[name]['age'] = int(top_speakers[name]['age'])
                if top_speakers[name]['age'] <= opt.old_age:
                    top_speakers[name]['age'] = 'young'
                else:
                    top_speakers[name]['age'] = 'old'

            if top_speakers[name]['religion'] in ['atheist', 'agnostic', 'secular']:
                top_speakers[name]['religion'] = 'atheist'

    empty_lines = 0
    vocab = defaultdict(lambda: 0)
    fl = [f for f in os.listdir(opt.in_dir) if f.endswith('_json_filtered_tokenized')]
    with open('java_ctx_embeds_reddit_demographic', 'w') as out_file:
        for file in tqdm(fl):
            with open(opt.in_dir + '/' + file) as handle:
                print('opening file ' + file)
                for line in handle.readlines():
                    try:
                        tline = json.loads(line)
                        author = tline['author']
                        text = tline['body']
                        words = text.split(' ')
                        for w_ in words:
                            vocab[w_] += 1

                        if len(text.strip()) == 0:
                            empty_lines += 1
                            continue

                        out_file.write(author \
                                    + '\t' + top_speakers[author]['age'] \
                                    + '\t' + top_speakers[author]['location'] \
                                    + '\t' + top_speakers[author]['religion'] \
                                    + '\t' + top_speakers[author]['gender'] \
                                    + '\t' + text + '\n')
                    except:
                        print('error in file: ' + file)
                        print('error with line: ' + line)
                        print('\n\n')

    print('Number of lines skipped because of empty text: ' + str(empty_lines))
    # OUT FORMAT: "0	age	location	religion	gender	words in the message"

    with open(opt.vocab, 'w') as out_file:
        for k,v in sorted(vocab.items(), key=operator.itemgetter(1), reverse=True):
            out_file.write(str(v) + '\t' + k + '\n')


if __name__ == '__main__':
    main()
