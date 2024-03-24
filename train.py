import numpy as np
from pathlib import Path

from numpy_vit import VisionTransformer
from dataset import ImageFolderDataset, DataLoader
from learning import softmax, cross_entropy_loss, accuracy, derivative_cross_entropy_softmax

if __name__ == "__main__":
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 2
    img_size = 64
    embed_dim = 256
    hidden_dim = 512
    num_channels = 3
    num_heads = 8
    num_layers = 4
    num_classes = 10
    patch_size = 8
    num_patches = 64
    dropout_rate = 0.1
    validate = True
    save_dir = Path(__file__).parent / "data" / "models"

    ds_train = ImageFolderDataset("./data/train/")
    ds_val = ImageFolderDataset("./data/val/")
    ds_test = ImageFolderDataset("./data/test/")
    train_loader = DataLoader(ds_train, batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size, shuffle=True)
    model = VisionTransformer(image_size=img_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                              num_channels=num_channels, num_heads=num_heads, num_layers=num_layers,
                              num_classes=num_classes, patch_size=patch_size, num_patches=num_patches,
                              dropout=dropout_rate, learning_rate=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_parsed = 0
        acc = 0
        for images, labels in train_loader:
            predictions = model.forward(images)

            loss = cross_entropy_loss(softmax(predictions), labels)
            tmp = accuracy(predictions, labels)
            N = labels.shape[0]
            acc = ((acc * num_parsed) + (tmp * N)) / (num_parsed + N)
            num_parsed += N

            epoch_loss += loss
            print("Train loss: ", loss, "Train acc: ", tmp)

            d_loss = derivative_cross_entropy_softmax(softmax(predictions), labels)
            gradients = model.backward(d_loss)

        print(f'Train Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}, Acc: {acc}')

        if validate:
            epoch_loss = 0
            num_parsed = 0
            acc = 0
            for images, labels in val_loader:
                predictions = model.forward(images)

                loss = cross_entropy_loss(softmax(predictions), labels)
                tmp = accuracy(predictions, labels)
                N = labels.shape[0]
                acc = ((acc * num_parsed) + (tmp * N)) / (num_parsed + N)
                num_parsed += N

                print("Val loss: ", loss, "Val acc: ", tmp)
                epoch_loss += loss

            print(f'Val Epoch {epoch + 1}, Loss: {epoch_loss / len(val_loader)}, Acc: {acc}')

    model.save(save_dir=save_dir)

    epoch_loss = 0
    num_parsed = 0
    acc = 0
    for images, labels in test_loader:
        predictions = model.forward(images)

        loss = cross_entropy_loss(softmax(predictions), labels)
        tmp = accuracy(predictions, labels)
        N = labels.shape[0]
        acc = ((acc * num_parsed) + (tmp * N)) / (num_parsed + N)
        num_parsed += N

        print("Test loss: ", loss, "Test acc: ", tmp)
        epoch_loss += loss

    print(f'Test Loss: {epoch_loss / len(val_loader)}, Acc: {acc}')

