

import math, sys, os
import matplotlib as mpl
# mpl.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from argparse import ArgumentParser

IDX_ORDER = ['age', 'gender', 'religion', 'location']
RELIGION_VALUES = ['atheist', 'christian', 'muslim', 'buddhist', 'hindu']
LOCATION_VALUES = ['usa', 'asia', 'oceania', 'uk', 'europe', 'africa', 'mexico', 'southam', 'canada']

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = '20'
# plt.rcParams['figure.facecolor'] = (0.7226, 0.8008, 0.8945)

def main():
    parser = ArgumentParser()
    parser.add_argument('-f', '--find', dest='find', help='Find and write the combined set.', default=False, action='store_true')
    parser.add_argument('-p', '--plot', dest='plot', help='Plot statistics.', default=False, action='store_true')
    parser.add_argument('-r', '--rmbots', dest='rmbots', help='Remove known bots from complete_authors file.', default=False, action='store_true')
    opt = parser.parse_args()

    if not (opt.find or opt.plot or opt.rmbots):
        print('Must select --find or --plot or --rmbots...')
        sys.exit(0)

    if opt.find:
        find()
    elif opt.rmbots:
        rmbots()
    elif opt.plot:
        plot()

def rmbots():
    kb = []
    with open('../known_bots') as handle:
        for line in handle.readlines():
            kb.append(line.strip())

    nbots = 0
    with open('../demographic/complete_authors_nobots', 'w') as out_handle:
        with open('../demographic/complete_authors') as in_handle:
            for line in in_handle.readlines():
                uname = line.split('\t')[0]
                if uname not in kb:
                    out_handle.write(line)
                else:
                    nbots += 1

    print('Bots removed from author file: ' + str(nbots))

def plot():
    udict = {}
    r_by_l = defaultdict(lambda: defaultdict(lambda: 0))
    g_by_l = defaultdict(lambda: defaultdict(lambda: 0))
    total_male = 0
    unk_loc = 0
    unk_gen = 0
    unk_age = 0
    unk_rel = 0
    with open('../demographic/complete_authors') as handle:
        for line in handle.readlines():
            tline = line.strip().split('\t')
            tuser = {}
            for i in range(len(IDX_ORDER)):
                tuser[IDX_ORDER[i]] = tline[i+1]
            tuser['posts'] = int(tline[5])
            # condense non-religious
            if tuser['religion'] in ['atheist', 'agnostic', 'secular']:
                tuser['religion'] = 'atheist'

            if tuser['religion'] == 'UNK':
                unk_rel += 1
            if tuser['gender'] == 'UNK':
                unk_gen += 1
            if tuser['location'] == 'UNK':
                unk_loc += 1
            if tuser['age'] == 'UNK':
                unk_age += 1

            # enforce some sampling constraints
            nunks = sum([1 if tline[i] == 'UNK' else 0 for i in range(1, len(tline))])
            # if (tuser['location'] == 'usa' and nunks <= 1) or tuser['location'] != 'usa':
            #     if (tuser['gender'] == 'male' and total_male < 150000) or tuser['gender'] != 'male':
                    # if tuser['gender'] == 'male':
                    #     total_male += 1 

            if nunks < 3 and nunks != 0:
                r_by_l[tuser['location']][tuser['religion']] += 1
                g_by_l[tuser['location']][tuser['gender']] += 1
                udict[tline[0]] = tuser

    gposts, aposts, rposts, lposts = 0, 0, 0, 0
    num_unks = defaultdict(lambda: 0)
    for user in udict:
        nu = sum([0 if udict[user][k] != 'UNK' else 1 for k in udict[user]])
        num_unks[nu] += 1
        if udict[user]['gender'] != 'UNK':
            gposts += udict[user]['posts']
        if udict[user]['location'] != 'UNK':
            lposts += udict[user]['posts']
        if udict[user]['religion'] != 'UNK':
            rposts += udict[user]['posts']
        if udict[user]['age'] != 'UNK':
            aposts += udict[user]['posts']
    print('Number Unknowns: ' + str(dict(num_unks)))

    print('Number of posts for gender: ' + '{:,}'.format(gposts))
    print('Number of posts for age: ' + '{:,}'.format(aposts))
    print('Number of posts for religion: ' + '{:,}'.format(rposts))
    print('Number of posts for location: ' + '{:,}'.format(lposts))

    age_dist = [int(udict[u]['age']) for u in udict if udict[u]['age'] != 'UNK' and int(udict[u]['age']) < 100]
    age_oob = sum([1 if int(udict[u]['age']) >= 100 else 0 for u in udict if udict[u]['age'] != 'UNK'])
    print('Number of ages over 100: ' + str(age_oob))

    print('Average Age: ' + str(sum(age_dist) * 1.0 / len(age_dist)))
    med_age = sorted(age_dist)[int(len(age_dist)/2)]
    print('Median Age: ' + str(med_age))

    # fig = plt.subplot(2, 3, 1)
    fig = sns.distplot([age_dist])
    plt.ylabel('Density')
    plt.xlabel('Age')
    plt.title('Age Distribution')
    plt.xticks([i*10 for i in range(11)])
    # fig.patch.set_facecolor((0.7226, 0.8008, 0.8945))
    plt.show()
    plt.clf()

    # plt.subplot(2, 3, 2)
    unk_keys = [i for i in range(1, 4)]
    y_vals = [num_unks[v] for v in unk_keys]
    plt.bar(unk_keys, y_vals)
    plt.ylabel('Count')
    plt.xlabel('Number of Unknown Demographic Values')
    plt.title('Number of Unknowns per User')
    plt.xticks([1, 2, 3])
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
    # for i, v in enumerate(y_vals):
    #     ax.text(v + 0.25, i - 3, str(v), color='black', fontweight='bold')
    plt.show()
    plt.clf()

    # plt.subplot(2, 3, 3)
    post_dist = [math.log(udict[u]['posts'], 10) for u in udict]
    print('Total Number of Posts: ' + '{:,d}'.format(sum([udict[u]['posts'] for u in udict])))
    sns.distplot([post_dist])
    plt.xlabel(r'$log_{10}$ Number of Posts')
    plt.xticks(range(10))
    plt.ylabel('Density')
    plt.title('Distribution of Posts per User')
    # ax = plt.gca()
    # ax.set_yscale('log')
    plt.show()
    plt.clf()

    # plt.subplot(2, 3, 4)
    religions = {RELIGION_VALUES[i]: sum([1 if udict[u]['religion'] == RELIGION_VALUES[i] else 0 for u in udict]) for i in range(len(RELIGION_VALUES))}
    rkeys = religions.keys()
    plt.bar(rkeys, [religions[r] for r in rkeys])
    plt.xlabel('Religion')
    plt.ylabel('Number of People')
    plt.title('Distribution of Religions')
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
    plt.show()
    plt.clf()

    # plt.subplot(2, 3, 5)
    # pdp_r = {RELIGION_VALUES[i]: sum([udict[u]['posts'] for u in udict if udict[u]['religion'] == RELIGION_VALUES[i]]) for i in range(len(RELIGION_VALUES))}
    # plt.pie([pdp_r[RELIGION_VALUES[i]] for i in range(len(RELIGION_VALUES))], labels=RELIGION_VALUES)
    past_v = [g_by_l[lv]['male'] for lv in LOCATION_VALUES]
    plt.bar([i for i in range(len(LOCATION_VALUES))], past_v)

    new_v = [g_by_l[lv]['female'] for lv in LOCATION_VALUES]
    plt.bar([i for i in range(len(LOCATION_VALUES))], new_v, bottom=past_v)

    plt.xticks([i for i in range(len(LOCATION_VALUES))], LOCATION_VALUES, rotation=45)
    plt.legend(['male', 'female'])
    plt.title('Number of Users per Location by Gender')
    plt.show()
    plt.clf()

    # plt.subplot(2, 3, 6)
    # pdp_l = {LOCATION_VALUES[i]: sum([udict[u]['posts'] for u in udict if udict[u]['location'] == LOCATION_VALUES[i]]) for i in range(len(LOCATION_VALUES))}
    # plt.pie([pdp_l[LOCATION_VALUES[i]] for i in range(len(LOCATION_VALUES))], labels=LOCATION_VALUES)
    past_v = [r_by_l[lv][RELIGION_VALUES[0]] for lv in LOCATION_VALUES]
    plt.bar([i for i in range(len(LOCATION_VALUES))], past_v)
    for j in range(1, len(RELIGION_VALUES)):
        new_v = [r_by_l[lv][RELIGION_VALUES[j]] for lv in LOCATION_VALUES]
        plt.bar([i for i in range(len(LOCATION_VALUES))], new_v, bottom=past_v)
        past_v = [past_v[k] + new_v[k] for k in range(len(new_v))]
    plt.xticks([i for i in range(len(LOCATION_VALUES))], LOCATION_VALUES, rotation=45)
    plt.legend(RELIGION_VALUES)
    plt.title('Number of Users per Location by Religion')
    plt.show()
    plt.clf()

    num_male = 0
    num_female = 0
    for u in udict:
        if udict[u]['gender'] == 'male':
            num_male += 1
        elif udict[u]['gender'] == 'female':
            num_female += 1
    print('Number of Female: ' + '{:,}'.format(num_female))
    print('Number of Male: ' + '{:,}'.format(num_male))

    print('Number of Unknown Locations: ' + '{:,}'.format(unk_loc) + ' -- ' + '{:.2f}'.format(unk_loc*100.0 / len(udict)) + '%')
    print('Number of Unknown Genders: ' + '{:,}'.format(unk_gen) + ' -- ' + '{:.2f}'.format(unk_gen*100.0 / len(udict)) + '%')
    print('Number of Unknown Ages: ' + '{:,}'.format(unk_age) + ' -- ' + '{:.2f}'.format(unk_age*100.0 / len(udict)) + '%')
    print('Number of Unknown Religions: ' + '{:,}'.format(unk_rel) + ' -- ' + '{:.2f}'.format(unk_rel*100.0 / len(udict)) + '%')

    # plt.tight_layout()
    # plt.show()

def find():
    print('Reading locations...')
    udict = defaultdict(lambda: defaultdict(lambda: 'UNK'))
    with open('../demographic/resolved_locations') as handle:
        for line in handle.readlines():
            tline = line.strip().split('\t')
            udict[tline[0]]['location'] = tline[1]

    print('Reading genders...')
    with open('../demographic/gender_list') as handle:
        for line in handle.readlines():
            # udict[line.strip()]['gender'] = 'male'
            tline = line.strip().split('\t')
            if len(tline) != 2:
                tline = (tline[0][:-4] + '\tmale').split('\t')
            if udict[tline[0]]['gender'] == 'UNK':
                udict[tline[0]]['gender'] = []
            udict[tline[0]]['gender'].append(tline[1])

    nums = []
    MIN_PERCENT = 0.20
    num_smallp = 0
    num_both = 0
    print('\n\n')
    for u in udict:
        v = udict[u]['gender']
        if type(v) == list:
            nums.append(len(v))
            if 'male' in v and 'female' in v:
                num_both += 1
                sper = sum([1 if v[i] == 'male' else 0 for i in range(len(v))]) * 1.0 / len(v)
                if sper > 0.5:
                    major = 'male'
                    sper = 1 - sper
                else:
                    major = 'female'
                # print('sper is ' + str(sper) + ' and the list size is ' + str(len(v)))
                if sper < MIN_PERCENT:
                    num_smallp += 1
                    udict[u]['gender'] = major
                else:
                    udict[u]['gender'] = 'UNK'
            elif 'male' in v:
                udict[u]['gender'] = 'male'
            elif 'female' in v:
                udict[u]['gender'] = 'female'
    print('Number of people with gender listed: ' + str(len(nums)))
    print('Average number of genders per person: ' + str(sum(nums) / len(nums)))
    print('Number of people with small percent of one gender: ' + str(num_smallp))
    print('Number with both genders listed: ' + str(num_both))
    print('\n\n')

    print('Reading religions...')
    rfiles = [i for i in os.listdir('../demographic') if i.startswith('religions_')]
    for rfile in rfiles:
        with open('../demographic/' + rfile) as handle:
            for line in handle.readlines():
                tline = line.strip().split('\t')
                udict[tline[0]]['religion'] = tline[2]

    print('Reading ages...')
    afiles = [i for i in os.listdir('../demographic') if i.startswith('ages_')]
    for afile in afiles:
        with open('../demographic/' + afile) as handle:
            for line in handle.readlines():
                tline = line.strip().split('\t')
                if udict[tline[0]]['age'] == 'UNK':
                    udict[tline[0]]['age'] = []
                udict[tline[0]]['age'].append(int(tline[2]))

    nums = []
    nums_t = []
    MAX_AGE = 100
    MIN_AGE = 13
    MAX_RANGE = 9
    num_resolved = 0
    num_reasonable = 0
    num_broken = 0
    print('\n\n')
    for u in udict:
        v = udict[u]['age']
        if type(v) == list:
            nums.append(len(v))
            v = [i for i in v if i >= MIN_AGE and i < MAX_AGE]
            nums_t.append(len(v))
            if len(v) == 1:
                udict[u]['age'] = v[0]
                num_resolved += 1
            elif len(v) == 0:
                udict[u]['age'] = 'UNK'
                num_broken += 1
            else:
                age_range = max(v) - min(v)
                if age_range <= MAX_RANGE:
                    udict[u]['age'] = int(min(v) + age_range/2)
                    num_resolved += 1
                    num_reasonable += 1
                else:
                    udict[u]['age'] = 'UNK'
                    num_broken += 1
    print('Number of people with ages: ' + str(len(nums)))
    print('Average number of ages: ' + str(sum(nums) / len(nums)))
    print('Average number of ages trimmed invalids: ' + str(sum(nums_t) / len(nums_t)))
    print('Number of ages resolved: ' + str(num_resolved))
    print('Number of reasonable ages given aging: ' + str(num_reasonable))
    print('Number of ages broken: ' + str(num_broken))
    print('\n\n')

    print('Reading post counts...')
    pdict = defaultdict(lambda: 1)
    with open('../demographic/ppl_all') as handle:
        for line in handle.readlines():
            tline = line.strip().split('\t')
            pdict[tline[0]] = int(tline[1])

    print('Trimming users...')
    to_del = []
    for u in udict:
        nunks = 0
        for i in IDX_ORDER:
            if udict[u][i] == 'UNK':
                nunks += 1

        # Filter people with less than 2 known variables -- 3 unknowns means only one known or none.
        if nunks >= 3:
            to_del.append(u)
    print('Number of users trimmed: ' + '{:,}'.format(len(to_del)))
    for u in to_del:
        del udict[u]
    if '[deleted]' in udict:
        del udict['deleted']

    with open('../demographic/complete_authors', 'w') as handle:
        for user in udict:
            outstr = user
            for i in range(len(IDX_ORDER)):
                outstr += '\t' + str(udict[user][IDX_ORDER[i]])
            outstr += '\t' + str(pdict[user]) + '\n'
            handle.write(outstr)

if __name__ == '__main__':
    main()
