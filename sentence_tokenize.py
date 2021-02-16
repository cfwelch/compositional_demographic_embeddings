

import unicodedata, socket, json, html, os

from os import path
from tqdm import tqdm
from argparse import ArgumentParser
#from nltk.tokenize import sent_tokenize
from stanfordnlp.server import CoreNLPClient

def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--speakers', dest='speakers', help='Name of the file containing the speakers.', default='top_speakers', type=str)
    parser.add_argument('-p', '--prefix', dest='prefix', help='Directory containing the files to tokenize.', default='all_posts', type=str)
    parser.add_argument('-pos', '--pos', dest='pos', help='Part of speech tag the sentences.', default=False, action='store_true')
    parser.add_argument('-part', '--part', dest='part', help='Part of the array to iterate over.', default=1, type=int)
    parser.add_argument('-of', '--of', dest='of', help='Number of parts to split the array into.', default=1, type=int)
    opt = parser.parse_args()

    print('Sentence tokenizer!')
    sentence_count = 0

    annotator_string = 'tokenize,ssplit'
    if opt.pos:
        annotator_string += ',pos'

    top_speakers = []
    with open(opt.speakers) as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)
    top_speakers.sort()

    part_len = int(len(top_speakers) / opt.of)
    top_speakers = top_speakers[(opt.part-1)*part_len:] if opt.part == opt.of else top_speakers[(opt.part-1)*part_len:opt.part*part_len]
    print('Tokenizing part ' + str(opt.part) + ' of ' + str(opt.of) + '...')

    print('Loading Stanford CoreNLP...')
    with CoreNLPClient(annotators=[annotator_string], properties={'tokenize.options': 'ptb3Escaping=False'}, timeout=60000, memory='16G') as client:
        print('Iterating over files...')
        for file in top_speakers:
            if not path.exists(opt.prefix + '/' + file + '_json_filtered'):
                print(file + ' does not exist, skipping...')
                continue
            elif path.exists(opt.prefix + '/' + file + '_json_filtered_tokenized'):
                print(file + ' has already been written, skipping...')
                continue

            out_file = open(opt.prefix + '/' + file + '_sts', 'w')
            out_json = open(opt.prefix + '/' + file + '_json_filtered_tokenized', 'w')
            print('Opening file ' + file + '...')
            with open(opt.prefix + '/' + file + '_json_filtered') as handle:
                lines = handle.readlines()
                for line in tqdm(lines):
                    tline = json.loads(line)
                    tbody = tline['body']
                    # print(tbody)
                    if tbody == '':
                        continue
                    # sentences = sent_tokenize(tbody)
                    # Use Stanford tokenizer
                    parsed = client.annotate(tbody)

                    sentence_count += len(parsed.sentence)
                    full_sent = []
                    full_pos = []
                    for sent in parsed.sentence:
                        the_tokens = [i.originalText.replace(' ', '') for i in sent.token]
                        the_sent = ' '.join(the_tokens)
                        if opt.pos:
                            the_poss = [i.pos.replace(' ', '') for i in sent.token]
                            the_pos = ' '.join(the_poss)
                            full_pos.append(the_pos)
                        assert len(the_sent.split(' ')) == len(sent.token)
                        # assert len(the_sent.split(' ')) == len(sent['pos'])
                        the_sent = the_sent.lower()
                        full_sent.append(the_sent)
                        # out_file.write(the_sent.encode('utf-8') + '\n')
                        out_file.write(the_sent + '\n')
                    full_sent = ' '.join(full_sent)
                    full_pos = ' '.join(full_pos)
                    tline['body'] = full_sent
                    tline['pos'] = full_pos
                    out_json.write(json.dumps(tline) + '\n')
            print('Number of sentences so far: ' + '{:,}'.format(sentence_count))
            out_json.close()
            out_file.close()

if __name__ == '__main__':
    main()
