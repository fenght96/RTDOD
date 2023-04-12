import xml.etree.ElementTree as ET
import os
def gen_txt():
    used_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', )
    unused_classes = ('diningtable', 'dog', 'horse',
                'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor',)
    for train in ['trainval', ]:
        xml_path = '/media/datb/data/VOCdevkit/VOC2007/Annotations/'
        with open(f'/media/datb/data/VOCdevkit/VOC2007/ImageSets/Main/{train}.txt') as f:
            xmls = f.readlines()
        xmls = [xml.strip() + '.xml' for xml in xmls]


        file_num = 0
        file_txt = open(f'/media/datb/data/VOCdevkit/VOC2007/ImageSets/Main/{train}_20_10.txt', 'w')

        for xml in xmls:
            flag = False
            tree = ET.parse(xml_path + xml.strip())
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in unused_classes:
                    flag = True
            if not flag:
                continue
            else:
                file_txt.write(xml.split('.')[0] + '\n')
                file_num += 1
        file_txt.close()
        print(f"{train} number:", file_num)

def readfiles():
    root =  '/media/datb/data/VOCdevkit/VOC2007/Gen_annotations/'
    annos = os.listdir(root)
    annos = [anno.strip().split('.')[0] for anno in annos]
    with open(f'/media/datb/data/VOCdevkit/VOC2007/ImageSets/Main/train_20_left.txt', 'w') as f:
        for anno in annos:
            f.write(anno + '\n')

readfiles()