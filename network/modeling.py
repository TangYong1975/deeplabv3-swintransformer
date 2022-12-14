from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import berniwal_swintransformer
from .backbone import microsoft_swintransformer
from .backbone import hrnetv2
from .backbone import xception
from .backbone import ghostnetv2
from .backbone import mobilenetv2_bubbliiiing

from torchvision.models import resnet
from torchvision.models import mobilenetv2
from torchvision.models import mobilenetv3
from torchvision.models import regnet
from torchvision.models import vgg
from torchvision.models import shufflenetv2

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name == "resnet18":
        replace_stride_with_dilation=[False, False, False]
        inplanes = 512
        low_level_planes = 64
    else:
        replace_stride_with_dilation=[False, False, True]
        inplanes = 2048
        low_level_planes = 256

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _nostride_dilate(m, dilate):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate//2, dilate//2)
                m.padding = (dilate//2, dilate//2)
        else:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    if backbone_name == "mobilenet_v2_bubbliiiing": 
        backbone = mobilenetv2_bubbliiiing.mobilenet_v2(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:4]
        #--------------------------------
        import functools
        total_idx  = len(backbone.features)
        down_idx   = [2, 4, 7, 14]
        if output_stride == 8:
            for i in range(down_idx[-2], down_idx[-1]):
                backbone.features[i].apply(
                    functools.partial(_nostride_dilate, dilate=2)
                )
            for i in range(down_idx[-1], total_idx):
                backbone.features[i].apply(
                    functools.partial(_nostride_dilate, dilate=4)
                )
        elif output_stride == 16:
            for i in range(down_idx[-1], total_idx):
                backbone.features[i].apply(
                    functools.partial(_nostride_dilate, dilate=2)
                )
        #--------------------------------
        backbone.high_level_features = backbone.features[4:-1]

        inplanes = 320
        low_level_planes = 24
    elif backbone_name == "mobilenet_v2": 
        backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:4]
        backbone.high_level_features = backbone.features[4:-1]

        inplanes = 320
        low_level_planes = 24
    elif backbone_name == "mobilenet_v3_small":    
        backbone = mobilenetv3.mobilenet_v3_small(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:6]
        backbone.high_level_features = backbone.features[6:-1]

        inplanes = 96
        low_level_planes = 40
    elif backbone_name == "mobilenet_v3_large":    
        backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:8]
        backbone.high_level_features = backbone.features[8:-1]
 
        inplanes = 160
        low_level_planes = 80
    else:        
        raise NotImplementedError(backbone_name)

    backbone.features = None
    backbone.avgpool = None
    backbone.classifier = None

    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_berniwal_swintransformer(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    if backbone_name == "berniwal_swintransformer_swin_t":
        backbone = berniwal_swintransformer.swin_t(num_classes=num_classes)
        inplanes = 768
        low_level_planes = 192
    elif backbone_name == "berniwal_swintransformer_swin_s":
        backbone = berniwal_swintransformer.swin_s(num_classes=num_classes)
        inplanes = 768
        low_level_planes = 192
    elif backbone_name == "berniwal_swintransformer_swin_b":
        backbone = berniwal_swintransformer.swin_b(num_classes=num_classes)
        inplanes = 1024
        low_level_planes = 256
    else:
        backbone = berniwal_swintransformer.swin_l(num_classes=num_classes)
        inplanes = 1536
        low_level_planes = 384

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'stage2': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_microsoft_swintransformer(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]
   
    if backbone_name == "microsoft_swintransformer_swin_t":
        backbone = microsoft_swintransformer.swin_t(num_classes=num_classes, img_size=448) # img_size == crop_size
        inplanes = 768
        low_level_planes = 192
    elif backbone_name == "microsoft_swintransformer_swin_s":
        backbone = microsoft_swintransformer.swin_s(num_classes=num_classes, img_size=448)
        inplanes = 768
        low_level_planes = 192
    elif backbone_name == "microsoft_swintransformer_swin_b":
        backbone = microsoft_swintransformer.swin_b(num_classes=num_classes, img_size=448)
        inplanes = 1024
        low_level_planes = 256
    else:
        backbone = microsoft_swintransformer.swin_l(num_classes=num_classes, img_size=448)
        inplanes = 1536
        low_level_planes = 384
    
    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'stage2': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'forward_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    #backbone = IntermediateLayerGetter(backbone, return_layers=return_layers) # for Microsoft

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    
    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_regnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]
           
    if backbone_name == "regnet_y_400mf":
        backbone = regnet.regnet_y_400mf(pretrained=pretrained_backbone)
        inplanes = 440
        low_level_planes = 104
    elif backbone_name == "regnet_y_8gf":
        backbone = regnet.regnet_y_8gf(pretrained=pretrained_backbone)
        inplanes = 2016
        low_level_planes = 448
    elif backbone_name == "regnet_y_32gf":
        backbone = regnet.regnet_y_32gf(pretrained=pretrained_backbone)
        inplanes = 3712
        low_level_planes = 696
    else:
        raise NotImplementedError(backbone_name)

    backbone.low_level_features = backbone.trunk_output[0:2]
    backbone.high_level_features = backbone.trunk_output[2:4]
    backbone.trunk_output = None
    backbone.avgpool = None
    backbone.fc = None
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_vggnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    if backbone_name == "vgg11_bn": 
        backbone = vgg.vgg11_bn(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:15]
        backbone.high_level_features = backbone.features[15:-1]
    elif backbone_name == "vgg16_bn":    
        backbone = vgg.vgg16_bn(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:24]
        backbone.high_level_features = backbone.features[24:-1]
    elif backbone_name == "vgg19_bn":    
        backbone = vgg.vgg19_bn(pretrained=pretrained_backbone)
        backbone.low_level_features = backbone.features[0:27]
        backbone.high_level_features = backbone.features[27:-1]
    else:
        raise NotImplementedError(backbone_name)
    
    backbone.features = None
    backbone.avgpool = None
    backbone.classifier = None
    inplanes = 512
    low_level_planes = 256
   
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_shufflenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    if backbone_name == "shufflenet_v2_x0_5": 
        backbone = shufflenetv2.shufflenet_v2_x0_5(pretrained=pretrained_backbone)
        inplanes = 192
        low_level_planes = 48
    elif backbone_name == "shufflenet_v2_x1_0":    
        backbone = shufflenetv2.shufflenet_v2_x1_0(pretrained=pretrained_backbone)
        inplanes = 464
        low_level_planes = 116
    else:
        raise NotImplementedError(backbone_name)
    
    backbone.conv5 = None
    backbone.fc = None

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'stage2': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_ghostnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    if backbone_name == "ghostnet_v2_1_0":         
        backbone = ghostnetv2.ghostnet_v2_1_0(num_classes)
        inplanes = 160
        low_level_planes = 40
    elif backbone_name == "ghostnet_v2_1_3":         
        backbone = ghostnetv2.ghostnet_v2_1_3(num_classes)
        inplanes = 208
        low_level_planes = 52
    elif backbone_name == "ghostnet_v2_1_6":         
        backbone = ghostnetv2.ghostnet_v2_1_6(num_classes)
        inplanes = 256
        low_level_planes = 64
    else:
        raise NotImplementedError(backbone_name)

    backbone.low_level_features = backbone.blocks[0:5]
    backbone.high_level_features = backbone.blocks[5:-1]

    backbone.blocks = None
    backbone.global_pool = None
    backbone.conv_head = None
    backbone.act2 = None
    backbone.classifier = None
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone.startswith('mobilenet'):
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('berniwal_swintransformer'):
        model = _segm_berniwal_swintransformer(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('microsoft_swintransformer'):
        model = _segm_microsoft_swintransformer(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('regnet'):
        model = _segm_regnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('vgg'):
        model = _segm_vggnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('shufflenet'):
        model = _segm_shufflenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('ghostnet'):
        model = _segm_ghostnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3

def deeplabv3_resnet18(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet18', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet_v2_bubbliiiing(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model('deeplabv3', 'mobilenet_v2_bubbliiiing', num_classes, output_stride=16, pretrained_backbone=False)

def deeplabv3_mobilenet_v2(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenet_v2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet_v3_small(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'mobilenet_v3_small', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet_v3_large(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'mobilenet_v3_large', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_berniwal_swintransformer_swin_t(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'berniwal_swintransformer_swin_t', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_berniwal_swintransformer_swin_s(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'berniwal_swintransformer_swin_s', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_berniwal_swintransformer_swin_b(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'berniwal_swintransformer_swin_b', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_berniwal_swintransformer_swin_l(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'berniwal_swintransformer_swin_l', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_microsoft_swintransformer_swin_t(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'microsoft_swintransformer_swin_t', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_microsoft_swintransformer_swin_s(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'microsoft_swintransformer_swin_s', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_microsoft_swintransformer_swin_b(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'microsoft_swintransformer_swin_b', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_microsoft_swintransformer_swin_l(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'microsoft_swintransformer_swin_l', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=True): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_regnet_y_400mf(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'regnet_y_400mf', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_regnet_y_8gf(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'regnet_y_8gf', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_regnet_y_32gf(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'regnet_y_32gf', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_vgg11_bn(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'vgg11_bn', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_vgg16_bn(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'vgg16_bn', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_vgg19_bn(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'vgg19_bn', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_shufflenet_v2_x0_5(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'shufflenet_v2_x0_5', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_shufflenet_v2_x1_0(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'shufflenet_v2_x1_0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_shufflenet_v2_x2_0(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'shufflenet_v2_x2_0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_ghostnet_v2_1_0(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'ghostnet_v2_1_0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_ghostnet_v2_1_3(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'ghostnet_v2_1_3', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_ghostnet_v2_1_6(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3', 'ghostnet_v2_1_6', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# Deeplab v3+

def deeplabv3plus_resnet18(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet18', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet_v2_bubbliiiing(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'mobilenet_v2_bubbliiiing', num_classes, output_stride=16, pretrained_backbone=False)

def deeplabv3plus_mobilenet_v2(num_classes=21, output_stride=8, pretrained_backbone=False):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenet_v2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mobilenet_v3_small(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'mobilenet_v3_small', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mobilenet_v3_large(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'mobilenet_v3_large', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_berniwal_swintransformer_swin_t(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'berniwal_swintransformer_swin_t', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_berniwal_swintransformer_swin_s(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'berniwal_swintransformer_swin_s', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_berniwal_swintransformer_swin_b(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'berniwal_swintransformer_swin_b', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_berniwal_swintransformer_swin_l(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'berniwal_swintransformer_swin_l', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_microsoft_swintransformer_swin_t(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'microsoft_swintransformer_swin_t', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_microsoft_swintransformer_swin_s(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'microsoft_swintransformer_swin_s', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_microsoft_swintransformer_swin_b(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'microsoft_swintransformer_swin_b', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_microsoft_swintransformer_swin_l(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'microsoft_swintransformer_swin_l', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=True): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_regnet_y_400mf(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'regnet_y_400mf', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_regnet_y_8gf(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'regnet_y_8gf', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_regnet_y_32gf(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'regnet_y_32gf', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_vgg11_bn(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'vgg11_bn', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_vgg16_bn(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'vgg16_bn', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_vgg19_bn(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'vgg19_bn', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_shufflenet_v2_x0_5(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'shufflenet_v2_x0_5', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_shufflenet_v2_x1_0(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'shufflenet_v2_x1_0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_shufflenet_v2_x2_0(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'shufflenet_v2_x2_0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_ghostnet_v2_1_0(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'ghostnet_v2_1_0', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_ghostnet_v2_1_3(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'ghostnet_v2_1_3', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_ghostnet_v2_1_6(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'ghostnet_v2_1_6', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
