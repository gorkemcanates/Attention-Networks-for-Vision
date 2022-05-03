import torchvision.transforms as transforms

class Transforms:
    def __init__(self,
                 resize=256,
                 shape=224):
        self.train_transforms = transforms.Compose([transforms.RandomResizedCrop(shape),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ]
                                                   )


        self.val_transforms = transforms.Compose([transforms.Resize(resize),
                                                    transforms.CenterCrop(shape),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                  ]
                                                 )

class CIFARTransforms:
    def __init__(self):

        self.train_transforms = transforms.Compose([
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ]
                                                    )

        self.val_transforms = transforms.Compose([
                                                    # transforms.Resize((32, 32)),
                                                    # transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ]
                                                    )


