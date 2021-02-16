

import operator, random, copy, json, os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from colorutils import Color, random_rgb
from lexicon_map import *
from collections import defaultdict
from argparse import ArgumentParser

from utils import map_pos, liwc_keys, POS_DSET, POS_LSET

WORD_DIST_DIR = 'word_dists_bamman/user'
os.makedirs(WORD_DIST_DIR, exist_ok=True)
cat_cache = {}

def main():
    parser = ArgumentParser()
    parser.add_argument('-sp', '--sample_people', dest='sample_people', help='Randomly sample people for the graph.', default=-1, type=int)
    parser.add_argument('-ws', '--window_size', dest='window_size', help='Sliding window size of words to capture.', default=500, type=int)
    parser.add_argument('-agg', '--aggregate_stats', dest='aggregate_stats', help='Use aggregate statistics instead of per-person statistics.', default=True, action='store_false')
    parser.add_argument('-mp', '--map_pos_tags', dest='map_pos_tags', help='Map part-of-speech tags to the condensed categories.', default=False, action='store_true')
    parser.add_argument('-s', '--show', dest='show', help='Show the graph instead of saving it.', default=False, action='store_true')
    parser.add_argument('-z', '--skip_zeros', dest='skip_zeros', help='This skips all zero distance entries in a vocab and stretches vocabs to fit the window instead of lining them all up with zero entries for words that are not used by some users.', default=False, action='store_true')
    parser.add_argument('-o', '--only_use', dest='only_use', help='Only use N users instead of all speakers.', default=-1, type=int)
    parser.add_argument('-l', '--lastn', dest='lastn', help='Use the N most different words from each speakers vocab.', default=5000, type=int)
    parser.add_argument('-u', '--use_keys', dest='use_keys', help='Can be POS, LIWC, or ROGET and defaults to POS.', default='POS', type=str)
    parser.add_argument('-sb', '--skip_by', dest='skip_by', help='How many words to move when sliding the window.', default=1, type=int)
    opt = parser.parse_args()

    pos_map = {}
    with open('pos_dist_out_above_95') as handle:
        for line in handle:
            parts = line.strip().split(' --- ')
            pos_map[parts[0]] = parts[1]

    def get_pos_tag(word):
        rv = 'NONE'
        if word in pos_map:
            rv = pos_map[word]
        if opt.map_pos_tags and opt.use_keys == 'POS':
            rv = map_pos(rv)
        return rv

    top_speakers = []
    with open('top_speakers') as handle:
        for line in handle:
            tline = line.strip().split('\t')[0]
            top_speakers.append(tline)
    if opt.only_use > 0:
        top_speakers = top_speakers[:opt.only_use]

    token_counts = {ts: defaultdict(lambda: 0) for ts in top_speakers}
    all_tcount = 1
    for tspeak in top_speakers:
        with open('vocabs/user/' + tspeak + '_vocab') as handle:
            for line in handle.readlines():
                tline = line.split('\t')
                if len(tline) != 2:
                    # print(tspeak + ': ' + str(tline))
                    continue
                wcount = int(tline[1].strip())
                token_counts[tspeak][tline[0].strip()] = wcount
                all_tcount += wcount

    if opt.sample_people > 0:
        random.shuffle(top_speakers)
        top_speakers = top_speakers[:opt.sample_people]

    key_set = POS_LSET
    if opt.map_pos_tags and opt.use_keys == 'POS':
        key_set = POS_DSET
    if opt.use_keys == 'LIWC':
        key_set = copy.deepcopy(liwc_keys)
    elif opt.use_keys == 'ROGET':
        key_set = copy.deepcopy(roget_keys)
    key_set.append('NONE')

    affect_x = defaultdict(lambda: 0)
    affect_y = defaultdict(lambda: defaultdict(lambda: []))
    vocab_lens = []
    for user in tqdm(top_speakers):
        with open(WORD_DIST_DIR + '/' + user + '_word_dist', encoding='utf-8', errors='ignore') as handle:
            counter = 0
            window = defaultdict(lambda: [])
            win_weight = []
            win_tcount = []
            all_lines = handle.readlines()
            for line in tqdm(all_lines[-opt.lastn:]): # Take the last N words for the window
                parts = line.split('\t')
                if len(parts) != 2:
                    print('Line does not have 2 parts: \'' + line + '\'')
                float_sim = float(parts[0])
                if float_sim == 0.0 and opt.skip_zeros:
                    continue
                wordstr = parts[1].strip()

                if len(win_tcount) >= opt.window_size:
                    win_tcount = win_tcount[-opt.window_size:]
                # print(user + ' says ' + wordstr + ' with freq ' + str(token_counts[user][wordstr]) + ' and similarity ' + str(float_sim))
                win_weight.append(1.0) # float_sim * (token_counts[user][wordstr] * 1.0 / all_tcount)
                if len(win_weight) >= opt.window_size:
                    win_weight = win_weight[-opt.window_size:]

                lcats = get_lcats(wordstr, get_pos_tag, opt.use_keys)
                for CAT_TYPE in key_set:
                    if CAT_TYPE == 'NONE':
                        continue
                    if CAT_TYPE in lcats:
                        window[CAT_TYPE].append(CAT_TYPE)
                    else:
                        if len(lcats) > 0:
                            window[CAT_TYPE].append('NULL')
                        else:
                            window[CAT_TYPE].append('NONE')
                    # print(window)
                    if len(window[CAT_TYPE]) >= opt.window_size:
                        window[CAT_TYPE] = window[CAT_TYPE][-opt.window_size:]
                        affect_y[user][CAT_TYPE].append(sum([win_weight[i] for i in range(len(window[CAT_TYPE])) if window[CAT_TYPE][i] == CAT_TYPE]))
                        if CAT_TYPE == key_set[0]:
                            affect_y[user]['NONE'].append(sum([win_weight[i] for i in range(len(window[CAT_TYPE])) if window[CAT_TYPE][i] == 'NONE']))
                            affect_x[counter] += 1
                counter += 1
            vocab_lens.append(counter)
    print('Vocab Lengths: ' + str(vocab_lens))

    graph_directory = WORD_DIST_DIR + '_bamman_plots'
    graph_directory += '_agg' if opt.aggregate_stats else '_per'
    graph_directory += '_' + opt.use_keys.lower()
    if opt.use_keys == 'POS' and not opt.map_pos_tags:
        graph_directory += '_all'
    
    if not os.path.exists(graph_directory):
        os.makedirs(graph_directory)

    # score_dict = {i: {} for i in range(4)}
    score_dict = defaultdict(lambda: 0)
    for CAT_TYPE in key_set:
        ax = plt.subplot(1, 1, 1) # 1,2,1
        cat_score = gen_plot(opt, affect_x, affect_y, CAT_TYPE, top_speakers, opt.skip_by)
        ax.set_title(CAT_TYPE + ' Words in Sliding Window (N=' + str(opt.window_size) + ') Over Vocab Sorted by Similarity')

        for k in range(len(cat_score)):
            # score_dict[k][CAT_TYPE] = cat_score[k]
            score_dict[CAT_TYPE] += cat_score[k]

        # for user in top_speakers:
        #     affect_y[user].reverse()

        # ax = plt.subplot(1, 2, 2)
        # gen_plot(affect_x, affect_y, top_speakers)
        # ax.set_title('Different ' + CAT_TYPE + ' Words')
        if opt.show:
            plt.show()
        else:
            plt.savefig(graph_directory + '/' + CAT_TYPE + '.png')
        plt.clf()

    # for q in range(len(score_dict)):
    #     print('\n\nLooking at section ' + str(q+1) + '...')
    #     for k,v in sorted(score_dict[q].items(), key=operator.itemgetter(1), reverse=True):
    #         print(k + ': ' + str(v))

    for k,v in sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True):
        print(k + ': ' + str(v))

def get_lcats(wordstr, get_pos_tag, usekeys):
    retval = None
    if wordstr not in cat_cache:
        if usekeys == 'LIWC':
            lcats = get_liwc_parts(wordstr)
        elif usekeys == 'ROGET':
            lcats = get_roget_parts(wordstr)
        elif usekeys == 'POS':
            lcats = get_pos_tag(wordstr)
        cat_cache[wordstr] = lcats
        retval = lcats
    else:
        retval = cat_cache[wordstr]
    return retval

def gen_plot(opt, affect_x, affect_y, cat_type, top_speakers, skip_by):
    ax_axis = [i-(opt.window_size-1) for i in affect_x.keys()]
    affect_y = copy.deepcopy(affect_y)
    print('Converting points to percentages for cat_type ' + str(cat_type) + '...')
    for user in tqdm(top_speakers):
        tay = []
        for i in range(len(ax_axis)):
            tindex = int(i * 1.0 / len(ax_axis) * len(affect_y[user][cat_type]))
            # print('tindex is ' + str(tindex))
            # print('userlen: ' + str(len(affect_y[user])))
            tay.append(affect_y[user][cat_type][tindex] * 100.0 / opt.window_size)
        affect_y[user][cat_type] = tay

        # SKIP BY
        t_affect_y = []
        for i in range(0, len(affect_y[user][cat_type]), skip_by):
            t_affect_y.append(affect_y[user][cat_type][i])
        affect_y[user][cat_type] = t_affect_y
    ax_axis = range(len(affect_y[user][cat_type]))


        # affect_y[user].extend([0]*(len(ax_axis) - len(affect_y[user])))
    # print('afy: ' + str(affect_y[user][cat_type]))

    # ay_avg = [np.average([affect_y[user][cat_type][i] for user in top_speakers if affect_y[user][cat_type][i] > 0]) for i in range(len(ax_axis))]
    # ay_std = [np.std([affect_y[user][cat_type][i] for user in top_speakers if affect_y[user][cat_type][i] > 0]) for i in range(len(ax_axis))]
    # num_users = [np.sum([1 for user in top_speakers if affect_y[user][cat_type][i] > 0]) for i in range(len(ax_axis))]
    # ay_above = np.array(ay_avg) + np.array(ay_std)
    # ay_below = np.array(ay_avg) - np.array(ay_std)

    ay_avg = []
    ay_above = []
    ay_below = []
    sum_bins = [0, 0, 0, 0]
    for i in range(len(ax_axis)):
        tarr = [affect_y[user][cat_type][i] for user in top_speakers if affect_y[user][cat_type][i] > 0]
        # ay_avg.append(np.average(tarr))
        # print(tarr)
        if tarr == []:
            ay75, ay50, ay25 = 0, 0, 0
        else:
            ay75, ay50, ay25 = np.percentile(tarr, [75, 50, 25])
        # print('ay75: ' + str(ay75))
        # print('ay50: ' + str(ay50))
        # print('ay25: ' + str(ay25))
        # print('np.average: ' + str(np.average(tarr)))
        ay_above.append(ay75)
        ay_below.append(ay25)
        ay_avg.append(ay50)
        sum_bins[int(i/(len(ax_axis)/len(sum_bins)))] += ay50

    color_fill = Color(random_rgb())
    color_edge = color_fill - Color((16, 16, 16))

    x_min = min(ax_axis)
    x_max = max(ax_axis)

    if opt.aggregate_stats:
        plt.plot(ax_axis, ay_avg, c=[cc*1.0/255 for cc in color_edge.rgb], label='Median')
        # plt.plot(ax_axis, ay_above, c=np.random.rand(3,), label='+std')
        # plt.plot(ax_axis, ay_below, c=np.random.rand(3,), label='-std')
        plt.fill_between(ax_axis, ay_below, ay_above, alpha=0.5, edgecolor=color_edge.hex, facecolor=color_fill.hex)
        # plt.plot(ax_axis, num_users, c='black', label='Number of Users')
        print('median line: ' + ''.join(['(' + str(xiz[0]) + ',' + str(xiz[1]) + ')' for xiz in zip(ax_axis, ay_avg)]))
        print('iqr3: ' + ''.join(['(' + str(xiz[0]) + ',' + str(xiz[1]) + ')' for xiz in zip(ax_axis, ay_above)]))
        print('iqr1: ' + ''.join(['(' + str(xiz[0]) + ',' + str(xiz[1]) + ')' for xiz in zip(ax_axis, ay_below)]))
    else:
        for user in top_speakers:
            plt.plot(ax_axis, affect_y[user][cat_type], c=np.random.rand(3,))

    plt.xlabel('Distance of Sliding Window from Most Similar Words')
    plt.ylabel('Percentage of Words in Category') 
    plt.xticks([x_min, (x_min + x_max) / 4, (x_min + x_max) / 2, (x_min + x_max) / 4 * 3, x_max], ['0%', '25%', '50%', '75%', '100%'])
    plt.legend()
    return sum_bins

if __name__ == '__main__':
    main()
