from torchvision.datasets import ImageFolder

class CTDataset:
    def __init__(self, root_dir, transform=None, num_classes=2):
   
        self.data = ImageFolder(root=root_dir, transform=transform) # With ImageFolder, each folder is a class
        # Number of classes for the one hot encoder
        self.num_classes = num_classes

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
    
        img, label = self.data[idx]
        
    #    label_one_hot = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return img, label