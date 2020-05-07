import random
import numpy as np
from collections import defaultdict
import time
import torch
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from data_helper import LabeledDataset
from helper import collate_fn, compute_ts_road_map
from models import *
import pdb


def loss_function(x_hat, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        x_hat, x.float()
    )
    KLD = 0.5 * torch.mean(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + KLD


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_ts = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            threat_score = 0.0
            loss_tot = 0.0

            for i, (samples, targets, road_images, extras) in enumerate(dataloaders[phase]):
                inputs = torch.stack(samples).to(device)
                labels = torch.stack(road_images).to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, mu, logvar = model(inputs)
                    loss = loss_function(outputs, labels, mu, logvar)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        for output, label in zip(outputs, labels):
                            output = output.detach().cpu().numpy()
                            label = label.detach().cpu().numpy()
                            ts_road_map = compute_ts_road_map(output > 0.5, label)
                            threat_score += ts_road_map
                loss_tot += loss.item()
            if phase == 'train':
                print('train loss: {}'.format(loss_tot / len(labeled_trainset)))
            elif phase == 'val':
                print('val loss: {}'.format(loss_tot / len(labeled_valset)))
                epoch_ts = threat_score.item() /len(labeled_valset)
                print('ts score: {}'.format(epoch_ts))
        if epoch_ts > best_ts:
            print("saving best model at epoch {}".format(epoch))
            best_ts = epoch_ts
            torch.save(model.state_dict(), "models_pkl/ResNetVAE_0507_main.pkl")
            # torch.save(optimizer.state_dict(), "models_pkl/ResNetVAE_optimizer_0507.pkl")
            # torch.save(scheduler.state_dict(), "models_pkl/ResNetVAE_scheduler_0507.pkl")

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    epoch = 100
    batchsize = 16
    num_class = 1
    threshold = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'

    labeled_scene_index = np.arange(106, 130)
    labeled_scene_index_val = np.arange(130, 134)

    transform_train = transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ])
    transform_val = transforms.ToTensor()

    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform_train,
                                      extra_info=True
                                      )
    train_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batchsize, shuffle=True, num_workers=4,
                                              collate_fn=collate_fn)

    labeled_valset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=labeled_scene_index_val,
                                    transform=transform_val,
                                    extra_info=True
                                    )
    val_loader = torch.utils.data.DataLoader(labeled_valset, batch_size=batchsize, shuffle=True, num_workers=4,
                                            collate_fn=collate_fn)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    model = ResNetVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    train_model(model, optimizer, lr_scheduler, num_epochs=epoch)