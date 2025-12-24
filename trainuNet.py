import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from uNetUtils import CustomDataset, uNet
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="params for training")

    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batchsize", type=int, default=2, help="batch size, depends on GPU VRAM")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--stepsize", type=int, default=2, help="frequency step size for learning reduction")

    return parser.parse_args()


def main():


    args = get_args()

    print(f"Epochs configured: {args.epochs}")
    print(f"Batch size configured: {args.batchsize}")
    print(f"Initial learning rate configured: {args.lr}")
    print(f"Step Size optimizer configured: {args.stepsize}")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)


    ### Dataset Preparation ###

    image_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])


    img_path = "dataset/train/"
    mask_path = "dataset/train_masks/"


    dataset = CustomDataset(img_dir=img_path,mask_dir=mask_path,img_transform=image_transform)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.6,0.4], generator=generator)

    batch_size = args.batchsize

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=2)


    model = uNet(in_channels=3, num_classes=1)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr= args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.stepsize, gamma=0.9)


    epochs = args.epochs

    for epoch in range(epochs):
        model.train()
        train_running_loss = 0
        
        print("training")

        for index, (img,mask) in enumerate(tqdm(train_loader)):

            img = img.float().to(device)
            mask = mask.float().to(device)

            prediction = model(img)

            optimizer.zero_grad()
            loss = criterion(prediction, mask)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()


        train_loss = train_running_loss / (index + 1)

        model.eval()
        val_running_loss = 0

        with torch.no_grad():

            print("validating")

            for index, (img,mask) in enumerate(tqdm(val_loader)):
                
                img = img.float().to(device)
                mask = mask.float().to(device)

                prediction = model(img)

                loss = criterion(prediction,mask)
                val_running_loss += loss.item()

            val_loss = val_running_loss / (index + 1)


        print(f'Epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, lr: {optimizer.param_groups[0]["lr"]}')

        scheduler.step()

    torch.save(model.state_dict(), "uNetModel.pth")
    print("modelo guardado")




if __name__ == "__main__":
    main()
