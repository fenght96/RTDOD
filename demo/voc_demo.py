# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
import xml.etree.ElementTree as ET
from lxml import etree

# constants
WINDOW_NAME = "COCO detections"
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
            )

 
class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
 
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
 
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
 
        child3 = etree.SubElement(self.root, "source")
 
        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child5.text = "Unknown"
 
        child6 = etree.SubElement(child3, "image")
        child6.text = "flickr"
        child7 = etree.SubElement(child3, "flickrid")
        child7.text = "35435"
 
 
    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)
    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')
    def add_pic_attr(self,label,xmin,ymin,xmax,ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)





def readxml(path):
    
    bboxes = []
    classes = []
    tree = ET.parse(path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if CLASSES.index(name) >= 10:
            classes.append(name)
            bbox = obj.find("bndbox")
            bboxes.append([
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ])
    return bboxes, classes



def writexml(path, predictions, bb, cls):
    print(predictions)
    bboxes = predictions['instances'].get('pred_boxes')
    scores = predictions['instances'].get('scores')
    classes = predictions['instances'].get('pred_classes')
    h,w = predictions['instances'].image_size
    

    filename = path.split('/')[-1]
    anno = GEN_Annotations(filename)

    anno.set_size(w, h, 3)
    for i in range(len(classes)):
        box = bboxes.__getitem__(i)
        class_i = classes[i]
        class_i = CLASSES[class_i.item()]
        for b in box:
            xmin, ymin, xmax, ymax = b
            anno.add_pic_attr(class_i, xmin.item(),ymin.item(),xmax.item(),ymax.item())
    for i in range(len(bb)):
        b = bb[i]
        class_i = cls[i]
        anno.add_pic_attr(class_i, b[0],b[1],b[2],b[3])
    anno.savefile(path.replace('JPEGImages', 'Gen_annotations'))



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    from projects.SparseRCNN.sparsercnn import add_sparsercnn_config
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        root = '/media/datb/data/VOCdevkit/VOC2007/JPEGImages/'
        args.input = '/media/datb/data/VOCdevkit/VOC2007/ImageSets/Main/trainval_20_10.txt'
        with open(args.input, 'r') as f:
            lines = f.readlines()
        lines = [os.path.join(root, line.strip() + '.jpg') for line in lines]


        for path in tqdm.tqdm(lines):
            # use PIL, to be consistent with evaluation
#             img = read_image(path, format="BGR")
            # SparseRCNN uses RGB input as default 
            
            img = read_image(path, format="RGB")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            bb, cls = readxml(path.replace('jpg', 'xml').replace('JPEGImages', 'Annotations'))
            # print(f"bb:{bb}\n cls:{cls}")
            writexml(path.replace('jpg', 'xml'),
                predictions,
                bb, cls)

            

            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            # if cv2.waitKey(0) == 27:
            #     break  # esc to quit
            
                
    
