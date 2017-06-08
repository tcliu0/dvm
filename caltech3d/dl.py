import os
import requests
from xml.etree import ElementTree

base_url = 'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/?delimiter=/&prefix='
res_url = 'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/'
dirs = ['Image_Datasets/3D_objects/ImageSet070504/',
        'Image_Datasets/3D_objects/ImageSet071204/',
        'Image_Datasets/3D_objects/ImageSet071404/',
        'Image_Datasets/3D_objects/ImageSet071604/',
        'Image_Datasets/3D_objects/ImageSet072104/',
        'Image_Datasets/3D_objects/ImageSet072204/',
        'Image_Datasets/3D_objects/ImageSet072304/',
        'Image_Datasets/3D_objects/ImageSet072604/',
        'Image_Datasets/3D_objects/ImageSet072704/',
        'Image_Datasets/3D_objects/ImageSet072804/',
        'Image_Datasets/3D_objects/ImageSet072904/',
        'Image_Datasets/3D_objects/ImageSet080304/',
        'Image_Datasets/3D_objects/ImageSet080404/',
        'Image_Datasets/3D_objects/ImageSet091504/']

ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

def crawl(url):
    r = requests.get(url)
    root = ElementTree.fromstring(r.content)

    for child in root.findall('s3:CommonPrefixes', ns):
        next = child.find('s3:Prefix', ns).text
        print next
        dir = next.split(os.path.sep)[-2]
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)
        crawl(base_url + next)
        os.chdir('..')

    for child in root.findall('s3:Contents', ns):
        key = child.find('s3:Key', ns).text
        print key
        fname = key.split(os.path.sep)[-1]
        r = requests.get(res_url + key)
        f = open(fname, 'wb')
        f.write(r.content)
        f.close()

for d in dirs:
    dir = d.split(os.path.sep)[-2]
    if not os.path.exists(dir):
        os.mkdir(dir)
    os.chdir(dir)
    crawl(base_url + d)
    os.chdir('..')
