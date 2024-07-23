import os
import shutil
import random
import torch
import torchvision
from PIL import Image
import zipfile
import socket
import sys
import ssl

def prepare_test_set(root_dir, class_names):
    test_dir = os.path.join(root_dir, 'test')
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    
    os.mkdir(test_dir)
    for class_name in class_names:
        os.mkdir(os.path.join(test_dir, class_name))
    
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if os.path.exists(class_dir):
            images = [x for x in os.listdir(class_dir) if x.lower().endswith('png')]
            selected_images = random.sample(images, min(30, len(images)))
            for image in selected_images:
                source_path = os.path.join(class_dir, image)
                target_path = os.path.join(test_dir, class_name, image)
                shutil.copy(source_path, target_path)
    
    return test_dir

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = []
            for dir_path in image_dirs[class_name]:
                if os.path.exists(dir_path):
                    images += [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images

        self.images = {}
        self.class_names = ['Normal', 'Viral Pneumonia', 'COVID']
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
        self.transform = transform

    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])

    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_path = self.images[class_name][index]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

def train_and_evaluate_model(dl_train, dl_test, class_names, model_file_name, epochs=1):
    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=len(class_names))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)
    print(resnet18)

    print('Starting training..')
    for e in range(0, epochs):
        print('=' * 20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('=' * 20)
        train_loss = 0.
        val_loss = 0.
        resnet18.train()  # set model to training phase
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 20 == 0:
                print('Evaluating at step', train_step)
                accuracy = 0
                resnet18.eval()  # set model to eval phase
                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())
                val_loss /= (val_step + 1)
                accuracy = accuracy / len(dl_test.dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                resnet18.train()
                if accuracy >= 0.95:
                    print('Performance condition satisfied, stopping..')
                    torch.save(resnet18.state_dict(), model_file_name)
                    return
        train_loss /= (train_step + 1)
        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')
    torch.save(resnet18.state_dict(), model_file_name)

def main():
    model_file_name = ""
    socket_path = ""

    if os.path.exists(sys.argv[1]):
        model_file_name = sys.argv[1]
        socket_path = ''
    else:
        model_file_name = ''
        socket_path = sys.argv[1]

    class_names = ['Normal', 'Viral Pneumonia', 'COVID']

    # Combine datasets from multiple hospitals
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    hospital_path = '/tmp/work'
    if not os.path.isdir(hospital_path):
        os.mkdir(hospital_path, mode=0o777)
    
    hospitals = []
    for data_dir in sys.argv[2:]:
        if zipfile.is_zipfile(data_dir):
            with zipfile.ZipFile(data_dir, 'r') as zip_ref:
                zip_ref.extractall(hospital_path)
            hospitals.append(os.path.join(hospital_path, os.path.basename(data_dir).replace('.zip', '')))
        else:
            hospitals.append(data_dir)

    # Prepare combined training dataset
    train_dirs = {class_name: [] for class_name in class_names}
    for hospital in hospitals:
        for class_name in class_names:
            train_dirs[class_name].append(os.path.join(hospital, class_name))

    train_image_dirs = {class_name: train_dirs[class_name] for class_name in class_names}
    train_dataset = ChestXRayDataset(train_image_dirs, train_transform)

    print(f'Total number of training images: {len(train_dataset)}')

    # Prepare test dataset
    test_dirs = {class_name: [] for class_name in class_names}
    for hospital in hospitals:
        test_dir = prepare_test_set(hospital, class_names)
        for class_name in class_names:
            test_dirs[class_name].append(os.path.join(test_dir, class_name))

    test_image_dirs = {class_name: test_dirs[class_name] for class_name in class_names}
    test_dataset = ChestXRayDataset(test_image_dirs, test_transform)
    
    batch_size = 6
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print('Number of training batches:', len(dl_train))
    print('Number of test batches:', len(dl_test))

    train_and_evaluate_model(dl_train, dl_test, class_names, model_file_name, epochs=1)
    
    if socket_path != '':
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(socket_path)
            with open(model_file_name, 'rb') as f:
                data = f.read()
                client.sendall(data)        
        finally:
            client.close()

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
