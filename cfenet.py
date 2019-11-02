import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from layers.nn_utils import *


class CFENet(nn.Module):

    def __init__(self, phase, cfg, num_classes=21):
        super(CFENet, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.num_classes = num_classes
        self.cfe_config = self.cfg.CFENET_CONFIGS
        self.base = get_basemodel(self.cfg.backbone)
        self.Norm = nn.BatchNorm2d(self.cfe_config['channels'][0],
                                   eps=1e-5, momentum=0.01, affine=True)
        self._prepare_extras()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def _prepare_extras(self, ):
        self._prepare_ffb()
        self._prepare_arterial()
        self._prepare_lateral()
        self._prepare_LOC_CONF()

    def conv_dw(self,inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))

    def _prepare_ffb(self, ):
        first_out, second_out, third_out = self.cfg.backbone_out_channels
        self.backbone_out = second_out
        # self.reduce1 = BasicConv(first_out, self.cfe_config['channels'][0] // 2,
        #                          kernel_size=3, stride=1, padding=1)

        self.reduce1=self.conv_dw(first_out, self.cfe_config['channels'][0] // 2,stride=1)

        # self.up_reduce1 = BasicConv(second_out, self.cfe_config['channels'][0] // 2, kernel_size=1)
        self.reduce2 = BasicConv(second_out, self.cfe_config['channels'][1] // 2, kernel_size=1)
        self.up_reduce2 = BasicConv(third_out, self.cfe_config['channels'][1] // 2, kernel_size=1)

        # s1 = F.upsample(self.up_reduce1(t1), scale_factor=2, mode='bilinear')
        # self.upsample_convT = nn.ConvTranspose2d( self.cfe_config['channels'][0] // 2,\
        # self.cfe_config['channels'][0] // 2,kernel_size=20,stride=1,padding=0,bias=False)

    def _prepare_arterial(self, ):
        arterial = list()
        channels = [self.backbone_out] + self.cfe_config['channels'][1:]
        # print("_prepare_arterial channels:\n{}".format(channels))
        for i, channel in enumerate(channels):
            if i == len(channels) - 1:
                continue
            elif i == len(channels) - 2:
                if self.cfg.input_size == 300:
                    arterial.append(
                        nn.Conv2d(channels[i],
                                  channels[i + 1],
                                  kernel_size=3,
                                  stride=1,
                                  padding=0))
                elif self.cfg.input_size == 512:
                    arterial.append(
                        nn.Conv2d(channels[i],
                                  channels[i + 1],
                                  kernel_size=2,
                                  stride=1,
                                  padding=0))
            else:
                stride = 1 if i == 0 else 2
                arterial.append(
                    get_CFEM(cfe_type='normal',
                             in_planes=channels[i],
                             out_planes=channels[i + 1],
                             stride=stride,
                             scale=1,
                             groups=8)
                )
        # print("arterial:\n{}".format(arterial))
        self.arterial = nn.ModuleList(arterial)
        # print(" \nself arterial:\n{}".format(self.arterial))

    def _prepare_lateral(self, ):
        lateral = list()
        for i in range(self.cfe_config['lat_cfes']):
            lateral.append(
                get_CFEM(cfe_type='large',
                         in_planes=self.cfe_config['channels'][i],
                         out_planes=self.cfe_config['channels'][i],
                         stride=1,
                         scale=1,
                         groups=8,
                         dilation=2)
            )
        self.lateral = nn.ModuleList(lateral)

    def _prepare_LOC_CONF(self, ):
        loc, conf = list(), list()
        for i in range(self.cfe_config['maps']):
            loc.append(
                nn.Conv2d(self.cfe_config['channels'][i], self.cfe_config['ratios'][i] * 4, kernel_size=3, stride=1,
                          padding=1)
            )
            conf.append(
                nn.Conv2d(self.cfe_config['channels'][i], self.cfe_config['ratios'][i] * self.num_classes,
                          kernel_size=3, stride=1, padding=1)
            )
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

    def forward_base(self, x):
        if self.cfg.backbone == 'vgg':
            for k in range(23):
                x = self.base[k](x)
            t0 = x
            for k in range(23, len(self.base)):
                x = self.base[k](x)
            t1 = x

        elif self.cfg.backbone == 'seresnet50':
            t0, t1 = self.base(x)

        elif self.cfg.backbone == "mobilenet":
            for k in range(12):
                x = self.base[k](x)
            t0 = x
            for k in range(12, len(self.base)):
                x = self.base[k](x)
            t1 = x
        return t0, t1

    def forward_sources(self, x):
        t0, t1 = x
        # print("\n t0,t1:\n{}\t{}".format(t0.shape,t1.shape))
        sources = []
        # s0 = self.reduce1(t0)
        # s1 = F.upsample(self.up_reduce1(t1), scale_factor=2, mode='bilinear')
        # s1 = self.upsample_convT(self.up_reduce1(t1)) #model too big
        # s1=torch.reshape(t1,s0.shape)
        # print("\n s0,s1,s1':\n{}\t{}\t{}".format(s0.shape,self.up_reduce1(t1).shape,s1.shape))
        # s = torch.cat((s0, s1), 1)

        # modify
        s = t0
        # print("\n s :\n{}\t".format(s.shape))
        sources.append(self.lateral[0](s))
        s0 = self.reduce2(t1)
        x = t1
        for k, v in enumerate(self.arterial):
            if k == len(self.arterial) - 1:
                pass
            else:
                x = v(x)
                if k == 0:
                    s1 = self.up_reduce2(x)
                    # s1 = F.upsample(s1, scale_factor=2, mode='bilinear')
                    s = torch.cat((s0, s1), 1)
                    s = self.lateral[1](s)
                    # print("k:",k)
                    # print(" s0,s1:\n{}\t{}".format(s0.shape, s1.shape))
                    sources.append(s)
                else:
                    sources.append(x)
                    # print("k:",k)
                    # print("x:\n{}".format(x.shape))

        # print(" \n sources - - - -> \n")
        # for i in sources:
        # print(i.shape)
        return sources

    def init_model(self, base_model_path):

        # base_weights = torch.load(base_model_path)
        print('Loading base network...')
        # self.base.load_state_dict(base_weights)
        
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        
        self.base.apply(weights_init)
        print('Initializing weights...')
        self.arterial.apply(weights_init)
        self.lateral.apply(weights_init)
        self.Norm.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)
        self.reduce1.apply(weights_init)
        self.reduce2.apply(weights_init)

        # self.up_reduce1.apply(weights_init)
        self.up_reduce2.apply(weights_init)

        # self.upsample_convT.apply(weights_init)

    def forward(self, x):

        t0, t1 = self.forward_base(x)

        sources = self.forward_sources((t0, t1))
        loc, conf = [], []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)),
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_net(phase, cfg=None, num_classes=21):
    if phase not in ["train", "test"]:
        raise ValueError("Phase not recognized!")

    if cfg is None:
        raise ValueError("Please find cfg file in ./configs/* path")

    return CFENet(phase,
                  cfg=cfg,
                  num_classes=num_classes)


if __name__ == '__main__':
    cfgfile = 'configs/cfenet300_vgg16.py'
    from configs.CC import Config

    cfg = Config.fromfile(cfgfile)

    net = build_net('train', cfg.model, 21)
    print("build net:\n{}".format(net))
    # data = torch.autograd.Variable(torch.ones(2,3,512,512))
    # out = net(data)

    # print('Output: {}'.format([_.shape for _ in out]))
# Output: [torch.Size([2, 32720, 4]), torch.Size([2, 32720, 21])]

'''
CFENet(
  (base): ModuleList(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    (31): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
    (32): ReLU(inplace=True)
    (33): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
    (34): ReLU(inplace=True)
  )
  (Norm): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
  (reduce1): BasicConv(
    (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (up_reduce1): BasicConv(
    (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (reduce2): BasicConv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (up_reduce2): BasicConv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (arterial): ModuleList(
    (0): CFEM(
      (cfem_a): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (cfem_b): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(1024, 1024, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (ConvLinear): BasicConv(
        (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (shortcut): BasicConv(
        (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (relu): ReLU()
    )
    (1): CFEM(
      (cfem_a): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (cfem_b): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(1024, 1024, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (ConvLinear): BasicConv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (shortcut): BasicConv(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (relu): ReLU()
    )
    (2): CFEM(
      (cfem_a): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (cfem_b): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (ConvLinear): BasicConv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (shortcut): BasicConv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (relu): ReLU()
    )
    (3): CFEM(
      (cfem_a): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(64, 64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (cfem_b): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(64, 64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (ConvLinear): BasicConv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (shortcut): BasicConv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (relu): ReLU()
    )
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
  )
  (lateral): ModuleList(
    (0): CFEM(
      (cfem_a): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (cfem_b): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (ConvLinear): BasicConv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (shortcut): BasicConv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (relu): ReLU()
    )
    (1): CFEM(
      (cfem_a): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (cfem_b): ModuleList(
        (0): BasicConv(
          (conv): Conv2d(1024, 1024, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=8, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (1): BasicConv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=8, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        )
        (4): BasicConv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (ConvLinear): BasicConv(
        (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (shortcut): BasicConv(
        (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
      )
      (relu): ReLU()
    )
  )
  (loc): ModuleList(
    (0): Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (conf): ModuleList(
    (0): Conv2d(512, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(1024, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(512, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): Conv2d(256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): Conv2d(256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)

'''