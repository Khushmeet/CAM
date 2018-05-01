from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import json


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    normalize
])


def open_and_preprocess(image_path):
    img = Image.open(image_path)
    img.save('img.jpg')
    img = preprocess(img)
    img = Variable(img.unsqueeze(0))
    return img


with open('imagenet_classes.txt', 'r') as f:
    j = json.loads(f.read())
    classes = {int(key):value for (key, value) in j.items()}