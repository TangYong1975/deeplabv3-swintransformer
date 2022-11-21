# deeplabv3-swintransformer

from deeplabv3 (https://github.com/VainF/DeepLabV3Plus-Pytorch), and some swin-transformers (https://github.com/berniwal/swin-transformer-pytorch、https://github.com/microsoft/Swin-Transformer), I try these swin-transformers as backbone on deeplabv3, to see image segmentation and compare them.

note：
1. deeplabv3 and deeplabv3plus for berniwal swin-transformer;
2. deeplabv3 for microsoft swin-transformer;
3. deeplabv3plus for microsoft swin-transformer;
4. modify WindowAttention of berniwal swin-transformer by my thinking.
5. explanation by Issues 1
6. add dataset named 'CameraBase' for test
7. add hrnet and exception, follow VainF
8. add resnet18
9. add regnet (y_400mf, y_8gf, y_32gf), swin-transformer into (swin_t, swin_s, swin_b, swin_l), use resnet and mobilenet from pytorch
10. add mobilenet_v3(small, large)
11. add vggnet(vgg11_bn, vgg16_bn, vgg19_bn)
12. add shufflenetv2(x0_5, x1_0)
13. result(resnet50) for voc
