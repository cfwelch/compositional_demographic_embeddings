

known_bots = []
with open('known_bots') as handle:
    for line in handle.readlines():
        known_bots.append(line.strip().lower())

top_speakers = []
with open('top_speakers') as handle:
    for line in handle.readlines():
        top_speakers.append(line.strip().lower().split('\t')[0])

for kb in known_bots:
    if kb in top_speakers:
        print(kb)
