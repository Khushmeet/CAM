from torchvision import models
from torch.nn import functional as F
import numpy as np
import cv2
from args import parser
from utils import open_and_preprocess, classes


options = parser.parse_args()


def resnet152_cam(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


model = models.resnet152(pretrained=True)
final_layer = 'layer4'

model.eval()

features_blobs = []
model._modules.get(final_layer).register_forward_hook(hook_feature)
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

img = open_and_preprocess(options.img)
logit = model(img)
h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)

for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

cam = resnet152_cam(features_blobs[0], weight_softmax, [idx[0]])

print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
img = cv2.imread('img.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(
    cam[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM_output.jpg', result)
