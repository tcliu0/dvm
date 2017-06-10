import os
import re
import logging
from scipy import misc

def load_caltech3d_dataset():
    def loader(key):
        obj, view, lighting, angle = key
        path = 'caltech3d/processed/%s/%s' % (obj, view)
        files = os.listdir(path)
        start = 'img_1-%03d_' % (lighting*360 + angle)
        end = '_0_small.JPG'
        fname = [f for f in files if f.startswith(start) and f.endswith(end)]
        if not fname:
            print path
        fname = fname[0]
        im = misc.imread(os.path.join(path, fname)) / 255.0
        return im
    
    with open('caltech3d/train.txt') as f:
        train_dirs = [d.strip() for d in f.readlines()]
    train_data = []
    logging.info('Generating training tuples...')
    for obj in train_dirs:
        for view in ['top', 'bottom']:
            for lighting in range(3):
                for angle in range(5, 355, 5):
                    for delta in [20, 30, 40, 50, 60]:
                        if angle + delta <= 355:
                            i1 = (obj, view, lighting, angle)
                            i2 = (obj, view, lighting, angle + delta)
                            gt = (obj, view, lighting, angle + delta/2)
                            train_data.append((i1, i2, gt))

    with open('caltech3d/val.txt') as f:
        val_dirs = [d.strip() for d in f.readlines()]
    val_data = []
    logging.info('Generating validation tuples...')
    for obj in val_dirs:
        for view in ['top', 'bottom']:
            for lighting in range(3):
                for angle in range(5, 355, 5):
                    for delta in [20, 30, 40, 50, 60]:
                        if angle + delta <= 355:
                            i1 = (obj, view, lighting, angle)
                            i2 = (obj, view, lighting, angle + delta)
                            gt = (obj, view, lighting, angle + delta/2)
                            val_data.append((i1, i2, gt))

    with open('caltech3d/test.txt') as f:
        test_dirs = [d.strip() for d in f.readlines()]
    test_data = []
    logging.info('Generating test tuples...')
    for obj in test_dirs:
        for view in ['top', 'bottom']:
            for lighting in range(3):
                for angle in range(5, 355, 5):
                    for delta in [20, 30, 40, 50, 60]:
                        if angle + delta <= 355:
                            i1 = (obj, view, lighting, angle)
                            i2 = (obj, view, lighting, angle + delta)
                            gt = (obj, view, lighting, angle + delta/2)
                            test_data.append((i1, i2, gt))

    logging.info('Generated %d training, %d test and %d val tuples', len(train_data), len(test_data), len(val_data))
    return (train_data, test_data, val_data, loader)

def load_shapenet_dataset():
    def loader(key):
        obj, elevation, angle, depth = key
        path = 'shapenet/%s' % obj
        files = os.listdir(path)
        if depth:
            fname = 'depth_%d_%d.png' % (elevation, angle)
        else:
            fname = 'img_%d_%d.png' % (elevation, angle)
        im = misc.imread(os.path.join(path, fname)) / 255.0
        return im
    
    with open('shapenet/train.txt') as f:
        train_dirs = [d.strip() for d in f.readlines()]
    train_data = []
    logging.info('Generating training tuples...')
    for obj in train_dirs:
        for elevation in [10, 30]:
            for angle in range(5, 355, 5):
                for delta in [20, 30, 40, 50, 60]:
                    if angle + delta <= 355:
                        i1 = (obj, elevation, angle, False)
                        i2 = (obj, elevation, angle + delta, False)
                        gt = (obj, elevation, angle + delta/2, False)
                        dm = (obj, elevation, angle + delta/2, True)
                        train_data.append((i1, i2, gt, dm))

    with open('shapenet/val.txt') as f:
        val_dirs = [d.strip() for d in f.readlines()]
    val_data = []
    logging.info('Generating validation tuples...')
    for obj in val_dirs:
        for elevation in [10, 30]:
            for angle in range(5, 355, 5):
                for delta in [20, 30, 40, 50, 60]:
                    if angle + delta <= 355:
                        i1 = (obj, elevation, angle, False)
                        i2 = (obj, elevation, angle + delta, False)
                        gt = (obj, elevation, angle + delta/2, False)
                        dm = (obj, elevation, angle + delta/2, True)
                        val_data.append((i1, i2, gt, dm))

    with open('shapenet/test.txt') as f:
        test_dirs = [d.strip() for d in f.readlines()]
    test_data = []
    logging.info('Generating test tuples...')
    for obj in test_dirs:
        for elevation in [10, 30]:
            for angle in range(5, 355, 5):
                for delta in [20, 30, 40, 50, 60]:
                    if angle + delta <= 355:
                        i1 = (obj, elevation, angle, False)
                        i2 = (obj, elevation, angle + delta, False)
                        gt = (obj, elevation, angle + delta/2, False)
                        dm = (obj, elevation, angle + delta/2, True)
                        test_data.append((i1, i2, gt, dm))

    logging.info('Generated %d training, %d test and %d val tuples', len(train_data), len(test_data), len(val_data))
    return (train_data, test_data, val_data, loader)

def load_dataset(dataset):
    if dataset == 'caltech3d':
        logging.info('Using dataset caltech3d')
        return load_caltech3d_dataset()
    elif dataset == 'shapenet':
        logging.info('Using dataset shapenet')
        return load_shapenet_dataset()
    else:
        raise ValueError
