import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms

MODEL_PATH = r'model_output\densenet_two_resize_299.model'
LABEL_MAP_PATH = 'two_label_map.json'
TEST_DIR_PATH = 'data/two_dataset'
TEST_DIR = 'val'

INPUT_SIZE = 224
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':

    # Load model
    model = torch.load(MODEL_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # load dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(TEST_DIR_PATH, x), data_transforms['val']) for x in [TEST_DIR]}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=4) for x in [TEST_DIR]}

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloaders_dict[TEST_DIR]:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')