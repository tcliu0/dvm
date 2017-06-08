import numpy as np

items = {}

# generate list.txt with "unzip -l ShapeNetCore.v2.zip > list.txt"
with open('list.txt') as f:
    for line in f:
        tokens = [t for t in line.split(' ') if t]
        if len(tokens) == 4:
            tokens = [t for t in tokens[3].strip().split('/') if t]
            if len(tokens) == 3:
                cat_id = tokens[1]
                obj_id = tokens[2]
                if cat_id not in items:
                    items[cat_id] = []
                items[cat_id].append(obj_id)

total = 0

items = items.items()
lens = np.array([len(v) for k, v in items])
num_cats = len(lens)
cumulative = [sum(lens[:i]) for i in range(len(lens))]
total = cumulative[-1]
samples = np.random.choice(total, size=(400, ), replace=False)

objs = []

for sample in samples:
    rest = lens[sample >= cumulative]
    cat = len(rest) - 1
    offs = sample - cumulative[cat]

    catid = items[cat][0]
    objid = items[cat][1][offs]
    objs.append((catid, objid))

with open('train.txt', 'w') as f:
    train_objs = objs[0:300]
    train_objs.sort(key=lambda x:x[0] + (32 * '0' + x[1])[-32:])
    for obj in train_objs:
        f.write('%s/%s\n' % obj)
        
with open('val.txt', 'w') as f:
    val_objs = objs[300:320]
    val_objs.sort(key=lambda x:x[0] + (32 * '0' + x[1])[-32:])
    for obj in val_objs:
        f.write('%s/%s\n' % obj)

with open('test.txt', 'w') as f:
    test_objs = objs[320:400]
    test_objs.sort(key=lambda x:x[0] + (32 * '0' + x[1])[-32:])
    for obj in test_objs:
        f.write('%s/%s\n' % obj)
        
# extract subset (in bash) 'while read line; do unzip ShapeNetCore.v2.zip "ShapeNetCore.v2/"$line"/*"; done < all.txt
with open('all.txt', 'w') as f:
    objs.sort(key=lambda x:x[0] + (32 * '0' + x[1])[-32:])
    for obj in objs:
        f.write('%s/%s\n' % obj)
