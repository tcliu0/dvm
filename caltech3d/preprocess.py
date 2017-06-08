import os
import subprocess

if not os.path.exists('processed'):
    os.mkdir('processed')

def traverse_dir(dir):
    dirlist.append(dir)    
    os.chdir(dir)
    if not any([os.path.isdir(d) for d in os.listdir('.')]):
        if not any([d.find('Calibration') == 0 for d in dirlist]):
            print os.path.sep.join(dirlist)
            outdir = '../' * len(dirlist) + 'processed/' + dirlist[-2].lower()
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = os.path.join(outdir, dirlist[-1].lower())
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            for img in [f for f in os.listdir('.') if f.endswith('.JPG')]:
                imgnew = img.split('.')
                imgnew[0] += '_small'
                imgnew = '.'.join(imgnew)
                print imgnew
                subprocess.call(['convert', img, '-crop', '1536x1536+256+0', '-resize', '256x256', imgnew])
                os.rename(imgnew, os.path.join(outdir, imgnew))
    else:
        for dir in [d for d in os.listdir('.') if os.path.isdir(d)]:
            traverse_dir(dir)
    dirlist.pop()
    os.chdir('..')
    
dirlist = []

for dir in os.listdir('.'):
    if dir.find('ImageSet') != 0:
        continue
    traverse_dir(dir)

            

