
import random, json, os
from tqdm import tqdm

NSPEAKERS = 100
LENGTHS = {'train': 10000, 'valid': 1000, 'test': 1000, 'aa': 1000}

def main():
    top_speakers = []
    with open('top_speakers') as handle:
        for line in handle.readlines():
            top_speakers.append(line.strip().split('\t')[0])
    top_speakers = top_speakers[:NSPEAKERS]

    for speaker in tqdm(top_speakers):
        slines = []
        with open('all_posts_ts/' + speaker + '_json_filtered_tokenized') as handle:
            for line in handle.readlines():
                tline = json.loads(line)
                slines.append(tline['body'])
        random.shuffle(slines)

        sofar = 0
        for t in LENGTHS:
            os.makedirs('aa/' + speaker, exist_ok=True)
            with open('aa/' + speaker + '/' + t + '.txt', 'w') as handle:
                for i in slines[sofar:sofar+LENGTHS[t]]:
                    handle.write(i + '\n')
                sofar += LENGTHS[t]

if __name__ == '__main__':
    main()
