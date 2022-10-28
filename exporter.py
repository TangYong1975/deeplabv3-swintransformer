import torch
from PIL import Image
from torchvision import transforms
import network

model = network.deeplabv3plus_mobilenet(num_classes=2, output_stride=16)

model.load_state_dict((torch.load('checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth'))["model_state"])
model.eval()

input_image = Image.open('datasets/data/CameraBase/VOCdevkit/VOC2012/JPEGImages/0001-0019.jpg')
input_image = input_image.convert("RGB")
mytransform = transforms.Compose([
    transforms.Resize([448, 448]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = mytransform(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

output = model(input_batch)
print(output.shape)

# NOK (swimtransformer), OK (mobilenet, resnet50)
torch_onnx_out = torch.onnx.export(model, input_batch, 'checkpoints/best_deeplabv3plus_mobilenet_voc_os16.onnx',
                    export_params=True,
                    verbose=True,
                    input_names=['label'],
                    output_names=["output"],
                    opset_version=11)