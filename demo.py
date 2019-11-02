import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from cfenet import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO
from data import CFENET_ANCHOR_PARAMS

parser = argparse.ArgumentParser(description='CFENet Testing')
parser.add_argument('-c', '--config', default='configs/cfenet300_vgg16.py', type=str)
parser.add_argument('-f', '--directory', default='imgs/', help='the path to demo images')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
parser.add_argument('--video', default="", type=str, help='videofile mode')
parser.add_argument('--cam', default=-1, type=int, help='camera device id')
parser.add_argument('--show', action='store_true', help='Whether to display the images')
#add
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO version')

args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       CFENet Demo Program                             |\n'
           ' ----------------------------------------------------------------------', ['yellow','bold'])

global cfg
cfg = Config.fromfile(args.config)

anchor_config = CFENET_ANCHOR_PARAMS['{}_{}'.format(args.dataset, cfg.model.input_size)]
# anchor_config = anchors(cfg)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
# net = build_net('test',
                # size = cfg.model.input_size,
                # config = cfg.model.CFENET_CONFIGS)

num_classes = getattr(cfg.model.num_classes, args.dataset)
net = build_net('test', 
                cfg = cfg.model,
                num_classes=num_classes)
# init_net(net, cfg, args.trained_model)

state_dict = torch.load(args.trained_model)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()


print_info('===> Finished constructing and loading model',['yellow','bold'])
net.eval()
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
_preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(cfg.model.num_classes.VOC, cfg.loss.bkg_label, anchor_config)

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(cfg.model.num_classes.VOC, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.num_classes.VOC)]
# cats = [_.strip().split(',')[-1] for _ in open('data/coco_labels.txt','r').readlines()]
cats = [_.strip().split(',')[-1] for _ in open('data/voc0712.txt','r').readlines()]
labels = tuple(['__background__'] + cats)

def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr : #or np.NAN in box
            continue
        cls_indx = int(cls_inds[i])
#        print("box:",box,np.nan in box)
        
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

im_path = args.directory
cam = args.cam
video = args.video
if cam >= 0:
    capture = cv2.VideoCapture(cam)
    video_path = './cam'
if video:
    while True:
        # video_path = input('Please enter video path: ')
        video_path = video
        capture = cv2.VideoCapture(video_path)
        if capture.isOpened():
            break
        else:
            print('No file!')
if cam >= 0 or video:
    video_name = os.path.splitext(video_path)
    print("video_name:",video_name)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out_video = cv2.VideoWriter(video_name[0] + '_CFENet.mp4', fourcc, capture.get(cv2.CAP_PROP_FPS), (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
im_iter = iter(im_fnames)
while True:
    if cam < 0 and not video:
        try:
            fname = next(im_iter)
        except StopIteration:
            break
        if 'CFENet' in fname: continue # ignore the detected images
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
    else:
        ret, image = capture.read()
        if not ret:
            cv2.destroyAllWindows()
            capture.release()
            break
    loop_start = time.time()
    w,h = image.shape[1],image.shape[0]
    img = _preprocess(image).unsqueeze(0)
    if cfg.test_cfg.cuda:
        img = img.cuda()
    scale = torch.Tensor([w,h,w,h])
    out = net(img)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0]*scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    print("scores：\t",scores)
    print("boxes：\t",boxes)
    allboxes = []
    for j in range(1, cfg.model.num_classes.VOC):
        inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist()+[j] for _ in c_dets])

    loop_time = time.time() - loop_start
#    print("allboxs:",allboxes)
    if allboxes:
        allboxes = np.array(allboxes)   
        boxes = allboxes[:,:4]
        scores = allboxes[:,4]
        cls_inds = allboxes[:,5]
        # print("fname:",fname.split("/")[-1].split('.')[0])
        print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
                ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
        fps = 1.0 / float(loop_time) if cam >= 0 or video else -1
        image = draw_detection(image, boxes, scores, cls_inds, fps)
        # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

        if image.shape[0] > 1100:
            image = cv2.resize(image,
                                 (int(1000. * float(image.shape[1]) / image.shape[0]), 1000))
    if args.show:
        cv2.imshow('test', image)
        if cam < 0 and not video:
            cv2.waitKey(1000)
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                out_video.release()
                capture.release()
                break
    if cam < 0 and not video:
        cv2.imwrite('{}_CFENet.jpg'.format(fname.split('.')[0]), image)
        # print("im:\n",image.shape)
    else:
        out_video.write(image)
