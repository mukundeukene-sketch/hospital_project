def main():
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from tqdm import tqdm

    # ======================================
    # CONFIG
    # ======================================
    TRAIN_DIR = "dataset/train"
    TEST_DIR = "dataset/test"

    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    # ======================================
    # Image Transform
    # ======================================

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            )
        ], p=0.5),

        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        ),

        transforms.RandomErasing(
            p=0.15,
            scale=(0.02,0.15)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ======================================
    # Load Complete Training Dataset
    # ======================================
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset
    from sklearn.model_selection import StratifiedShuffleSplit

    # ======================================
    # Load Complete Dataset
    # ======================================
    full_dataset = ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print("Classes:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    print(f"\nTotal Images = {len(full_dataset)}")
    print(f"Number of Classes = {num_classes}")

    # ======================================
    # Stratified Train / Validation Split
    # ======================================

    targets = full_dataset.targets

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, val_idx = next(sss.split(range(len(targets)), targets))


    # ======================================
    # Train / Validation Split
    # ======================================
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset

    # Training dataset
    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    # Validation dataset
    val_dataset = ImageFolder(
        TRAIN_DIR,
        transform=test_transform
    )

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)

    # Use validation transforms

    # ======================================
    # Test Dataset
    # ======================================
    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    # ======================================
    # DataLoaders
    # ======================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ======================================
    # Dataset Summary
    # ======================================
    print("\nDataset Loaded Successfully!")
    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", DEVICE)

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes
    )

    model = model.to(DEVICE)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Train only classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )


    def train_one_epoch(model, loader, criterion, optimizer):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        loop = tqdm(loader)

        for images, labels in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(
                loss=loss.item(),
                acc=100 * correct / total
            )

        return running_loss / len(loader), 100 * correct / total

    def validate(model, loader):

        model.eval()

        loss_total = 0

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss_total += loss.item()

                _, predicted = outputs.max(1)

                total += labels.size(0)

                correct += predicted.eq(labels).sum().item()

        return loss_total / len(loader), 100 * correct / total

    EPOCHS = 20

    best_acc = 0

    for epoch in range(EPOCHS):
        if epoch == 5:
            print("Unfreezing EfficientNet backbone...")

            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - 5
            )

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader
        )

        scheduler.step()

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
            }, "best_model.pth")

            print("Best Model Saved!")

    checkpoint = torch.load("best_model.pth", map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")




if __name__ == "__main__":
    main()

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))