model = dict(
    type = 'cfenet',
    input_size = 300,
    backbone = 'vgg',
    resume_net = True,
    pretrained = 'weights/vgg16_reducedfc.pth',
    CFENET_CONFIGS = {
        'maps': 6,
        'lat_cfes': 2,
        'channels': [512, 1024, 512, 256, 256, 256],
        'ratios': [6, 6, 6, 6, 4, 4],
    },
    backbone_out_channels = (512, 1024, 1024),
    rgb_means = (104, 117, 123),
    p = 0.6,
    num_classes = dict(
        VOC = 6,
        # VOC = 21,
        # COCO = 81, # for VOC and COCO
        COCO = 6, # for VOC and COCO
        ),
    save_eposhs = 5,
    
    #add
    anchor_config = dict(
        step_pattern = [8, 16, 32, 64, 107, 320],
        size_pattern = [0.08, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        ),

    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    # warmup = 5,
    warmup = 1,
    per_batch_size = 8,
    init_lr = 0.002,
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        # COCO = [90, 120, 140, 160],
        # VOC = [150, 200, 250, 300],

        # COCO = [9, 12, 14, 16],
        COCO = [6, 10, 14, 18],
        VOC = [20, 40, 60, 80],
        # VOC = [15, 20, 25, 30],
        # 1epoch : 12min ;  [5,8]epoch:0.002, [9,12]epoch:0.0002,[13,16]:0.00002;[17,20]:0.000002

       
        ),
    print_epochs = 10,
    num_workers= 8,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    # score_threshold = 0.1,
    score_threshold = 0.5,
    keep_per_class = 50,
    save_folder = 'eval'
    )

loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC = dict(
        train_sets = [('2007', 'trainval')],
        eval_sets = [('2007', 'test')],
        ),
    COCO = dict(
        train_sets = [('2017', 'train'), ('2017', 'valminusminival')],
        eval_sets = [('2017', 'minival')],
        test_sets = [('2017', 'test-dev')],
        )
    )

import os
home = os.path.expanduser("~")
VOCroot = os.path.join(home,"Desktop/mmdetection/data/VOCdevkit/")
COCOroot = os.path.join(home,"data/cdet/roadmark/roadmark_0929_littlerotate_voc/coco2017")