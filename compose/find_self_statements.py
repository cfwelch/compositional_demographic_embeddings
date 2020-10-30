

import random, socket, json, sys, os, re
import seaborn as sns
import pandas as pd

import matplotlib as mpl
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from stanfordnlp.server import CoreNLPClient
from argparse import ArgumentParser

DIR_LOC = ''
DIR_SET = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', dest='type', help='Can be gender, location, religion, degree, or age.', default='gender', type=str)
    parser.add_argument('-i', '--intersect', dest='intersect', help='Find users in set intersection.', default=False, action='store_true')
    parser.add_argument('-p', '--plot', dest='plot', help='Plot the subset.', default=False, action='store_true')
    opt = parser.parse_args()

    if opt.intersect:
        find_intersection()
    elif opt.plot:
        plot_intersection()

    else:
        if opt.type not in ['gender', 'location', 'age', 'religion', 'degree']:
            print('Type must be gender, location, religion, degree, or age...')
            sys.exit(0)

        print('Finding ' + opt.type + ' statements...')
        if opt.type == 'gender':
            read_all_gender()
        elif opt.type == 'location':
            read_all_location()
        elif opt.type == 'age':
            read_all_age()
        elif opt.type == 'religion':
            read_all_religion()
        elif opt.type == 'degree':
            read_all_degree()

def plot_intersection():
    users = defaultdict(lambda: {})
    ppl_dict = defaultdict(lambda: [])
    with open('../demographic/intersect_set_manual') as handle: #_manual
        for line in handle.readlines():
            tline = line.strip().split('\t')

            users[tline[0]]['posts'] = int(tline[1])
            ppl_dict['posts'].append(int(tline[1]))

            users[tline[0]]['location'] = tline[2]
            lstr = 'unknown'
            if tline[2] in ['usa', 'europe', 'mideast', 'asia', 'canada', 'mexico', 'oceania']: #africa, southam
                lstr = tline[2]
            ppl_dict['location'].append(lstr)

            users[tline[0]]['gender'] = tline[3]
            ppl_dict['gender'].append(tline[3])

            users[tline[0]]['religion'] = tline[4]
            rint = 0
            if tline[4] == 'christian':
                rint = 1
            elif tline[4] in ['atheist', 'agnostic', 'secular']:
                rint = 2
            elif tline[4] == 'hindu':
                rint = 3
            elif tline[4] == 'buddhist':
                rint = 4
            elif tline[4] == 'muslim':
                rint = 5
            # add noise for visual
            rint += random.random()/2 - 0.25
            ppl_dict['religion'].append(rint)

            users[tline[0]]['age'] = int(tline[5])
            ppl_dict['age'].append(min(int(tline[5]), 60))

    # m_dist = [users[i]['posts'] for i in users if users[i]['gender'] == 'male']
    # f_dist = [users[i]['posts'] for i in users if users[i]['gender'] == 'female']
    # b_dist = [users[i]['posts'] for i in users if users[i]['gender'] == 'both']

    # BINS = 25
    # plt.hist(m_dist, bins=BINS, alpha=0.5, label='male')
    # plt.hist(f_dist, bins=BINS, alpha=0.5, label='female')
    # plt.hist(b_dist, bins=BINS, alpha=0.5, label='both')
    # plt.yscale('log', nonposy='clip')
    # plt.legend(loc='upper right')
    # plt.xlabel('Number of Posts')
    # plt.ylabel('Number of People')
    # plt.title('Number of Posts by Gender')
    # plt.show()


    # c_dist = [users[i]['posts'] for i in users if users[i]['religion'] == 'christian']
    # a_dist = [users[i]['posts'] for i in users if users[i]['religion'] in ['secular', 'atheist', 'agnostic']]
    # h_dist = [users[i]['posts'] for i in users if users[i]['religion'] == 'hindu']
    # b_dist = [users[i]['posts'] for i in users if users[i]['religion'] == 'buddhist']
    # m_dist = [users[i]['posts'] for i in users if users[i]['religion'] == 'muslim']

    # BINS = 25
    # plt.hist(c_dist, bins=BINS, alpha=0.5, label='christian')
    # plt.hist(a_dist, bins=BINS, alpha=0.5, label='atheist')
    # plt.hist(h_dist, bins=BINS, alpha=0.5, label='hindu')
    # plt.hist(b_dist, bins=BINS, alpha=0.5, label='buddhist')
    # plt.hist(m_dist, bins=BINS, alpha=0.5, label='muslim')
    # # plt.yscale('log', nonposy='clip')
    # plt.legend(loc='upper right')
    # plt.xlabel('Number of Posts')
    # plt.ylabel('Number of People')
    # plt.title('Number of Posts by Religion')
    # plt.show()


    # sns.distplot(c_dist, hist=False, kde_kws={"shade": True}, label='christian') #color="g", 
    # sns.distplot(a_dist, hist=False, kde_kws={"shade": True}, label='atheist')
    # sns.distplot(h_dist, hist=False, kde_kws={"shade": True}, label='hindu')
    # sns.distplot(b_dist, hist=False, kde_kws={"shade": True}, label='buddhist')
    # sns.distplot(m_dist, hist=False, kde_kws={"shade": True}, label='muslim')
    # Plot a historgram and kernel density estimate
    # sns.distplot(d, color="m", ax=axes[1, 1])


    # mpg = sns.load_dataset("mpg")
    # print(mpg)
    # print(type(mpg))
    # mpg = mpg.to_dict()
    # sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight", sizes=(40, 400), alpha=.5, palette="muted", height=6, data=mpg)

    mpl.rcParams['legend.labelspacing'] = 1
    ppl = pd.DataFrame.from_dict(ppl_dict)
    g = sns.relplot(x='religion', y='age', hue='gender', size='posts', sizes=(50, 750), alpha=0.5, palette='muted', height=6, data=ppl, legend='brief', style='location')
    # g._legend.labelspacing = 3

    # plt.setp(axes, yticks=[])
    # plt.legend(loc='upper right')
    START_AX = 1 # 0 if you include 'unknown'

    plt.xlim(left=-0.5 + START_AX, right=5.5)
    plt.grid(which='major', axis='x', color='black')
    plt.grid(which='major', axis='y', linestyle='-')
    plt.grid(which='minor', axis='y', linestyle='--')

    ax = plt.gca()
    ax.set_xticks([i for i in range(START_AX, 6)], minor=True)
    ax.set_xticks([i+0.5 for i in range(START_AX, 6)], minor=False)
    ax.set_xticklabels([], minor=False)
    xtickls = ['unknown', 'christian', 'atheist', 'hindu', 'buddhist', 'muslim'][START_AX:] 
    ax.set_xticklabels(xtickls, minor=True)

    ax.set_yticks([0, 10, 20, 30, 40, 50, 60], minor=False)
    ax.set_yticks([5, 15, 25, 35, 45, 55], minor=True)
    ax.set_yticklabels(['0', '10', '20', '30', '40', '50', '60+'], minor=False)

    plt.show()
    # plt.tight_layout()

def find_intersection():
    users = defaultdict(lambda: {})

    # open location files
    print('Reading location files...')
    location_files = [file for file in os.listdir('demographic') if file.startswith('locations_')]
    for file in location_files:
        with open('../demographic/' + file) as handle:
            for line in handle.readlines():
                tline = line.strip().split('\t')
                if tline[0] == '[deleted]':
                    continue
                users[tline[0]]['location'] = tline[3]

    # open religion files
    print('Reading religion files...')
    religion_files = [file for file in os.listdir('demographic') if file.startswith('religions_')]
    for file in religion_files:
        with open('../demographic/' + file) as handle:
            for line in handle.readlines():
                tline = line.strip().split('\t')
                if tline[0] == '[deleted]':
                    continue
                users[tline[0]]['religion'] = tline[2]

    # open ages files
    print('Reading age files...')
    age_files = [file for file in os.listdir('demographic') if file.startswith('ages_')]
    for file in age_files:
        with open('../demographic/' + file) as handle:
            for line in handle.readlines():
                tline = line.strip().split('\t')
                if tline[0] == '[deleted]':
                    continue
                users[tline[0]]['age'] = tline[2]

    # open gender files
    print('Reading gender files...')
    with open('../demographic/females_all') as handle:
        for line in handle.readlines():
            tline = line.strip()
            if tline == '[deleted]':
                continue
            users[tline]['gender'] = 'female'

    with open('../demographic/males_all') as handle:
        for line in handle.readlines():
            tline = line.strip()
            if tline == '[deleted]':
                continue
            users[tline]['gender'] = 'male'

    with open('../demographic/both_all') as handle:
        for line in handle.readlines():
            tline = line.strip()
            if tline == '[deleted]':
                continue
            users[tline]['gender'] = 'both'

    # print stats
    print('Number of users with gender: ' + '{:,}'.format(len([i for i in users if 'gender' in users[i]])))
    print('Number of users with location: ' + '{:,}'.format(len([i for i in users if 'location' in users[i]])))
    print('Number of users with religion: ' + '{:,}'.format(len([i for i in users if 'religion' in users[i]])))
    print('Number of users with age: ' + '{:,}'.format(len([i for i in users if 'age' in users[i]])))

    inter_set = [i for i in users if 'gender' in users[i] and 'location' in users[i] and 'religion' in users[i] and 'age' in users[i]]
    # inter_set = [i for i in users if 'gender' in users[i] and 'age' in users[i]]
    print('Number of users with location, gender, religion, and age: ' + '{:,}'.format(len(inter_set)))

    # ppl_all has counts of posts per user except if count is one -- cuts file size
    pdict = defaultdict(lambda: 1)
    with open('../demographic/ppl_all') as handle:
        for line in handle.readlines():
            tline = line.strip().split(': ')
            pdict[tline[0]] = int(tline[1])

    i_dist = [pdict[i] for i in inter_set]
    i_dist.sort()

    print('\nIntersection Statistics\n' + '='*30)
    print('Number of posts: ' + '{:,d}'.format(sum(i_dist)))
    print('Number of speakers: ' + '{:,d}'.format(len(i_dist)))
    print('Min-Max posts: ' + '{:,d}'.format(min(i_dist)) + ' - ' + '{:,d}'.format(max(i_dist)))
    print('IQR of posts: ' + '{:,d}'.format(i_dist[int(len(i_dist)/4)]) + ' - ' + '{:,d}'.format(i_dist[int(len(i_dist)*3/4)]))

    with open('../demographic/intersect_set', 'w') as handle:
        for i in inter_set:
            rel_val = users[i]['religion'] if 'religion' in users[i] else 'unknown'
            loc_val = users[i]['location'] if 'location' in users[i] else 'unknown'
            handle.write(i + '\t' + str(pdict[i]) + '\t' + loc_val + '\t' + users[i]['gender'] + '\t' + rel_val + '\t' + users[i]['age'] + '\n')

# christian, muslim, secular, atheist, agnostic, hindu, buddhist
def read_all_religion():
    for cur_dir in DIR_SET:
        with open('../demographic/religions_' + cur_dir, 'w') as outh:
            print('\n\nReading ' + cur_dir + '...')
            file_list = [i for i in os.listdir(DIR_LOC + cur_dir) if re.match('RC_\d\d\d\d-\d\d$', i)]
            for file in file_list:
                with open(DIR_LOC + cur_dir + '/' + file) as handle:
                    for line in tqdm(handle.readlines()):
                        tline = json.loads(line)
                        author = tline['author']
                        body = tline['body'].lower().replace('\n', ' ')
                        subr = tline['subreddit'].lower()
                        if 'fiction' in subr:
                            continue

                        match_o = re.match('.*?(i am|i\'m) (a )?(christian|muslim|secular|atheist|agnostic|hindu|buddhist).*?', body)
                        if match_o:
                            outh.write(author + '\t' + subr + '\t' + match_o.groups()[2] + '\n')

def read_all_location():
    MANUAL_LOCATION = ['canada']

    print('Loading Stanford CoreNLP...')
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner'], timeout=60000, memory='16G') as client:

        for cur_dir in DIR_SET:
            with open('../demographic/locations_' + cur_dir, 'w') as outh:
                print('\n\nReading ' + cur_dir + '...')
                file_list = [i for i in os.listdir(DIR_LOC + cur_dir) if re.match('RC_\d\d\d\d-\d\d$', i)]
                for file in file_list:
                    tlines = open(DIR_LOC + cur_dir + '/' + file, 'rb').read().decode('utf-8', errors='ignore').split('\n')
                    for line in tqdm(tlines):
                        if line.strip() == '':
                            continue
                        try:
                            tline = json.loads(line)
                        except ValueError:
                            print('line: ' + line)
                            sys.exit(0)

                        body = tline['body'].replace('\n', ' ')
                        author = tline['author']
                        subr = tline['subreddit'].lower()
                        if 'fiction' in subr:
                            continue

                        if 'i am from' in body or \
                           'i\'m from' in body or \
                           'i live in' in body:
                            # parsed = proc.parse_doc(body)
                            parsed = client.annotate(body)
                            # pprint(parsed)
                            # for sentence in parsed['sentences']:
                            for sentence in parsed.sentence:
                                lemma_list = [ll.lemma.lower() for ll in sentence.token]
                                for lid in range(len(lemma_list)-3):
                                    loc_type = None
                                    if lemma_list[lid:lid+3] == ['i', 'be', 'from']:
                                        loc_type = 'i am from'
                                    elif lemma_list[lid:lid+3] == ['i', 'live', 'in']:
                                        loc_type = 'i live in'

                                    if loc_type != None:
                                        loc = []
                                        pid = lid+3
                                        # print('\n\ntokens: ' + str(sentence['tokens']))
                                        # print('lemmas2end: ' + str(sentence['lemmas'][pid:]))
                                        # print('ner2end: ' + str(sentence['ner'][pid:]))
                                        # print('pos2end: ' + str(sentence['pos'][pid:]))
                                        while pid < len(sentence.token) and (sentence.token[pid].ner == 'LOCATION' or \
                                                                            sentence.token[pid].pos.startswith('NN') or \
                                                                            sentence.token[pid].lemma == 'the' or \
                                                                            sentence.token[pid].lemma in MANUAL_LOCATION):
                                            loc.append(sentence.token[pid].word)
                                            pid += 1
                                        if loc == ['the'] or len(loc) == 0:
                                            continue
                                        # print(author + '\t' + subr + '\t' + ' '.join(loc))
                                        outh.write(author + '\t' + subr + '\t' + loc_type + '\t' + ' '.join(loc) + '\n')

def read_all_degree():
    print('Loading Stanford CoreNLP...')
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner'], timeout=60000, memory='16G') as client:

        for cur_dir in DIR_SET:
            with open('../demographic/degrees_' + cur_dir, 'w') as outh:
                print('\n\nReading ' + cur_dir + '...')
                file_list = [i for i in os.listdir(DIR_LOC + cur_dir) if re.match('RC_\d\d\d\d-\d\d$', i)]
                for file in file_list:
                    tlines = open(DIR_LOC + cur_dir + '/' + file, 'rb').read().decode('utf-8', errors='ignore').split('\n')
                    for line in tqdm(tlines):
                        if line.strip() == '':
                            continue
                        try:
                            tline = json.loads(line)
                        except ValueError:
                            print('line: ' + line)
                            sys.exit(0)

                        body = tline['body'].replace('\n', ' ')
                        author = tline['author']
                        subr = tline['subreddit'].lower()
                        if 'fiction' in subr:
                            continue

                        if 'i am studying' in body or \
                           'i\'m studying' in body or \
                           'i studied' in body:
                            parsed = client.annotate(body)
                            # pprint(parsed)
                            for sentence in parsed.sentence:
                                lemma_list = [ll.lemma.lower() for ll in sentence.token]
                                pos_list = [ll.pos for ll in sentence.token]
                                # print(body + '\n' + str(lemma_list) + '\n' + str(pos_list) + '\n')
                                # input()
                                for lid in range(len(lemma_list)-3):
                                    deg_type = None
                                    offset_str = 3
                                    if lemma_list[lid:lid+3] == ['i', 'be', 'study']:
                                        deg_type = 'i am studying'
                                    elif lemma_list[lid:lid+2] == ['i', 'study']:
                                        deg_type = 'i studied'
                                        offset_str = 2

                                    if deg_type != None:
                                        deg = []
                                        pid = lid+offset_str
                                        hasNN = False
                                        while pid < len(sentence.token) and len(deg) < 5 and \
                                             (sentence.token[pid].pos.startswith('NN') or \
                                             sentence.token[pid].lemma == 'the' or \
                                             not hasNN):
                                            if sentence.token[pid].pos.startswith('NN'):
                                                hasNN = True
                                            deg.append(sentence.token[pid].word)
                                            pid += 1
                                        if deg == ['the'] or len(deg) == 0:
                                            continue
                                        # print(author + '\t' + subr + '\t' + ' '.join(deg))
                                        outh.write(author + '\t' + subr + '\t' + deg_type + '\t' + ' '.join(deg) + '\n')

def read_all_age():
    for cur_dir in DIR_SET:
        with open('../demographic/ages_' + cur_dir, 'w') as outh:
            print('\n\nReading ' + cur_dir + '...')
            file_list = [i for i in os.listdir(DIR_LOC + cur_dir) if re.match('RC_\d\d\d\d-\d\d$', i)]
            for file in file_list:
                with open(DIR_LOC + cur_dir + '/' + file) as handle:
                    for line in tqdm(handle.readlines()):
                        tline = json.loads(line)
                        author = tline['author']
                        body = tline['body'].lower().replace('\n', ' ')
                        subr = tline['subreddit'].lower()
                        if 'fiction' in subr:
                            continue

                        match_o = re.match('.*?(i am|i\'m) (\\d+) (years|yrs|yr) old[^e].*?', body)
                        if match_o:
                            outh.write(author + '\t' + subr + '\t' + match_o.groups()[1] + '\n')

def read_all_gender():
    males = []
    females = []
    for cur_dir in DIR_SET:
        print('\n\nReading ' + cur_dir + '...')
        file_list = [i for i in os.listdir(DIR_LOC + cur_dir) if re.match('RC_\d\d\d\d-\d\d$', i)]
        for file in file_list:
            with open(DIR_LOC + cur_dir + '/' + file) as handle:
                for line in tqdm(handle.readlines()):
                    tline = json.loads(line)
                    body = tline['body'].lower().replace('\n', ' ')
                    subr = tline['subreddit'].lower()
                    if 'fiction' in subr:
                        continue
                    if 'i\'m a guy' in body or \
                       'i am a guy' in body or \
                       'i am a male' in body or \
                       'i\'m a male' in body or \
                       'i am a man' in body or \
                       'i\'m a man' in body or \
                       'i am a boy' in body or \
                       'i\'m a boy' in body or \
                       'i am male' in body or \
                       'i\'m male' in body:
                        males.append(tline['author'])
                    if 'i\'m a girl' in body or \
                       'i am a girl' in body or \
                       'i\'m a gal' in body or \
                       'i am a gal' in body or \
                       'i am a female' in body or \
                       'i\'m a female' in body or \
                       'i am a woman' in body or \
                       'i\'m a woman' in body or \
                       'i am female' in body or \
                       'i\'m female' in body:
                        females.append(tline['author'])

        print('Males: ' + str(len(males)))
        print('Females: ' + str(len(females)))
        print('Unique Males: ' + str(len(set(males))))
        print('Unique Females: ' + str(len(set(females))))

    set_males = list(set(males))
    set_females = list(set(females))

    print('\nMales: ' + str(len([m for m in set_males if m not in set_females])))
    print('\nFemales: ' + str(len([f for f in set_females if f not in set_males])))
    print('\nBoth: ' + str(len([b for b in set_males+set_females if b in set_males and b in set_females])))

    with open('../demographic/gender_list', 'w') as handle:
        for f in females:
            handle.write(f + '\tfemale\n')
        for m in males:
            handle.write(m + '\tmale\n')

if __name__ == '__main__':
    main()
