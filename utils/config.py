from torchvision import transforms

def get_config(dataset):
    if dataset == 'MNIST':
        return dict(
            name='MNIST',
            type='classification',
            num_classes=10,
            feature_transforms=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
                transforms.Normalize((0.1307), (0.3081)),
            ]),
            input_shape=(32, 32),
            # input_shape=(28, 28),
            grayscale=True,
            normalize=((0.1307,), (0.3081,)),
            label_type='onehot',
            metric='OneHotMetric',
            criterion='CrossEntropyLoss',
        )
    
    if dataset == 'EMNIST':
        return dict(
            name='EMNIST',
            type='classification',
            split='balanced',
            num_classes=47,
            feature_transforms=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
                transforms.Normalize((0.17510404,), (0.33323708,)),
            ]),
            input_shape=(32, 32),
            # input_shape=(28, 28),
            grayscale=True,
            normalize=((0.17510404,), (0.33323708,)),
            label_type='onehot',
            metric='OneHotMetric',
            criterion='CrossEntropyLoss',
        )
    
    if dataset == 'CIFAR10':
        return dict(
            name='CIFAR10',
            type='classification',
            num_classes=10,
            feature_transforms=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))
            ]),
            input_shape=(32, 32),
            grayscale=False,
            normalize=((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)),
            label_type='onehot',
            metric='OneHotMetric',
            criterion='CrossEntropyLoss',
        )
    
    if dataset == 'CIFAR100':
        return dict(
            name='CIFAR100',
            type='classification',
            num_classes=100,
            feature_transforms=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071598291397095, 0.4866936206817627, 0.44120192527770996), (0.2673342823982239, 0.2564384639263153, 0.2761504650115967))
            ]),
            input_shape=(32, 32),
            grayscale=False,
            normalize=((0.5071598291397095, 0.4866936206817627, 0.44120192527770996), (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)),
            label_type='onehot',
            metric='OneHotMetric',
            criterion='CrossEntropyLoss',
        )