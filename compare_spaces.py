

import operator, faiss, re
import numpy as np

from tqdm import tqdm
from gensim.models import Word2Vec,KeyedVectors
from train_word2vec import EpochLogger
from utils import top_reddits

SPLIT_TYPE = 'user' #subreddit
EMBED_DIR = '../contextual_embeddings/person_embeds_100'

def main():
    dimension = 100
    max_dist = 100.0
    knn = 10
    neighbor_cutoff = 1

    # m1 = Word2Vec.load('embeddings/e20_d100_s100kv1/word2vec_mc3_e20_d100.model')
    # m1 = Word2Vec.load('embeddings/e20_d300_s5m/word2vec_mc3_e20_d300.model')
    # m2 = Word2Vec.load('embeddings/e20_d300_s400k/word2vec_mc3_e20_d300.model')

    iter_set = []
    if SPLIT_TYPE == 'user':
        iter_set = []
        with open('top_speakers') as handle:
            for line in handle:
                tline = line.strip().split('\t')[0]
                iter_set.append(tline)
    else:
        iter_set = top_reddits


    m2 = KeyedVectors.load_word2vec_format(EMBED_DIR + '/MAIN_ctx_embeds', binary=False)
    for tspeak in iter_set:
        print(tspeak)
        m1 = KeyedVectors.load_word2vec_format(EMBED_DIR + '/' + tspeak + '_ctx_embeds', binary=False)

        # index = faiss.IndexFlatL2(dimension)
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        # index = faiss.GpuIndexFlatL2(res, dimension, flat_config)
        index = faiss.GpuIndexFlatIP(res, dimension, flat_config)

        print('Vectors type is: ' + str(type(m1.wv.vectors)))
        print('Vectors shape: ' + str(m1.wv.vectors.shape))
        index.add(m1.wv.vectors)

        # qv = m1.wv['test'].reshape(1, dimension)
        print('Creating array of vectors from model...')
        nplist = []
        for in_word in tqdm(m1.wv.index2word):
            nplist.append(m1.wv[in_word])
        qv = np.array(nplist)

        print('Using faiss to search the matrix...')
        D, I = index.search(qv, knn+1)
        # print('D: ' + str(D))
        # print('I: ' + str(I))

        print('Summing distances in second space...')
        dist_map = {}
        neighbor_map = {}
        for ii in tqdm(range(len(m1.wv.index2word))):
            cur_word = m1.wv.index2word[ii]
            if re.search(r'\d', cur_word):
                continue

            dist_sum1 = np.average(D[ii][1:])
            near_words = [m1.wv.index2word[nn] for nn in I[ii][1:]]

            # dists_s2 = [m2.wv.similarity(word, cur_word) if word in m2.wv and cur_word in m2.wv else max_dist for word in near_words]
            # dist_sum2 = np.sum(dists_s2)
            dists_s2 = [m2.wv.similarity(word, cur_word) for word in near_words if word in m2.wv and cur_word in m2.wv]
            #print('dists_s1: ' + str(D[ii][1:]))
            if len(dists_s2) < neighbor_cutoff:
                continue
            dist_sum2 = np.average(dists_s2)

            # print('Sum of distances in space 1: ' + str(dist_sum1))
            # print('Nearest words: ' + str(near_words))
            # print('The distances in space 2 are: ' + str(dists_s2))
            # print('Sum of distances in space 2: ' + str(dist_sum2))

            # convert both to [0,1]
            #dist_sum1 = (dist_sum1 + 1) / 2.0
            dist_sum2 = (dist_sum2 + 1) / 2.0

            if not (dist_sum1 > 0 and dist_sum2 > 0):
                print('dist_sum1: ' + str(dist_sum1))
                print('dist_sum2: ' + str(dist_sum2))
                print(cur_word + ' --- ' + str(near_words))

            neighbor_map[cur_word] = near_words
            dist_map[cur_word] = abs(dist_sum2 - dist_sum1)

        with open('word_dists/' + SPLIT_TYPE + '/' + tspeak + '_word_dist', 'w') as handle:
            for k,v in sorted(dist_map.items(), key=operator.itemgetter(1)):
                handle.write(str(v) + '\t' + str(k) + '\t' + ','.join(neighbor_map[k]) + '\n')

if __name__ == '__main__':
    main()
