
import random, json

from tqdm import tqdm
from termcolor import colored
from collections import defaultdict

DATA_DIR = ''
CUTOFF = 1000
FILE_TO_USE = '/2011/RC_2011-05'

def main():
    # get_sample()
    annotate_sample()

def annotate_sample():
    out = open('sample_i_am_a_annotated', 'w')
    tallies = defaultdict(lambda: 0)
    i = 0
    try:
        with open('sample_i_am_a') as handle:
            for line in handle.readlines():
                tline = line.strip().replace('\r', '')
                print(tline.replace('i am a', colored('i am a', 'red')).replace('i\'m a', colored('i\'m a', 'red')))
                x = input('Annotation (' + str(i) + '): ').strip()
                out.write(x + '\n')
                tallies[x] += 1
                i += 1
    except:
        print('Closing annotator...')
        for k,v in tallies.items():
            print(k + ': ' + str(v))

def get_sample():
    i_am = []
    i_am_a = []
    with open(DATA_DIR + FILE_TO_USE) as handle:
        lines = handle.readlines()
        for line in tqdm(lines):
            tline = json.loads(line)
            body = tline['body'].replace('\n', ' ').replace('\r', '').lower()

            if 'i am ' in body or 'i\'m ' in body:
                i_am.append(body)

            if 'i am a ' in body or 'i\'m a ' in body or 'i am an ' in body or 'i\'m an ' in body:
                i_am_a.append(body)

    random.shuffle(i_am)
    random.shuffle(i_am_a)

    with open('sample_i_am', 'w') as handle:
        for line in i_am[:CUTOFF]:
            handle.write(line + '\n')

    with open('sample_i_am_a', 'w') as handle:
        for line in i_am_a[:CUTOFF]:
            handle.write(line + '\n')

if __name__ == '__main__':
    main()
