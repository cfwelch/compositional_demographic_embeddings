

import os

fl = os.listdir('all_posts')

kb = []
with open('known_bots') as handle:
    for line in handle.readlines():
        kb.append(line.strip())

for k in kb:
    if k + '_json' in fl:
        torm = 'all_posts/' + k + '_json'
        os.remove(torm)
        print('Removed ' + torm)
    if k + '_json_filtered' in fl:
        torm = 'all_posts/' + k + '_json_filtered'
        os.remove(torm)
        print('Removed ' + torm)

