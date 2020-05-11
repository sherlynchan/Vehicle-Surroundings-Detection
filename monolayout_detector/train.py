
import numpy as np
import time
# import eval_segm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from validate import *
import os
import json
import tqdm
import argparse
from eval import *
# from utils import *
# from kitti_utils import *
# from layers import *
# from metric.iou import IoU
# from fcn_iou import *
from IPython import embed
from torch.autograd import Variable

from helper import collate_fn
import model
from data_loader_single_map import *

image_folder = '../../data'
annotation_csv = '../../data/annotation.csv'

unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 126)
labeled_scene_index_val = np.arange(126, 134)

def get_args():
    parser 	= argparse.ArgumentParser(description="MonoLayout options")
    parser.add_argument("--data_path", type=str, default="../../data",
                         help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="models/",
                         help="Path to save models")
    parser.add_argument("--model_name", type=str, default="monolayout",
                         help="Model Name with specifications")
    parser.add_argument("--ext", type=str, default="png",
                         help="File extension of the images")
    parser.add_argument("--height", type=int, default=512,
                         help="Image height")
    parser.add_argument("--width", type=int, default=512,
                         help="Image width")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="learning rate")
    parser.add_argument("--lr_D", type=float, default=1e-5,
                         help="discriminator learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                         help="step size for the both schedulers")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                         help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=128,
                         help="size of topview occupancy map")
    parser.add_argument("--num_epochs", type=int, default=100,
                         help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                         help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=12,
                         help="Number of cpu workers for dataloaders")
    parser.add_argument("--lambda_D", type=float, default=0.01,
                         help="tradeoff weight for discriminator loss")
    parser.add_argument("--discr_train_epoch", type=int, default=5,
                         help="epoch to start training discriminator")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                         help="OSM path")
    parser.add_argument("--pos", type=int, default=1,
                        help="pos")

    return parser.parse_args()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {}
        self.weight["dynamic"] = self.opt.dynamic_weight
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []

		# Initializing models
        self.models["encoder"] = model.Encoder(self.opt.height, self.opt.width, True)
        self.models["decoder"] = model.Decoder(self.models["encoder"].num_ch_enc)
        self.models["discriminator"] = model.Discriminator()

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(self.models[key].parameters())
            else:
                self.parameters_to_train += list(self.models[key].parameters())

        # Optimization 
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 
            self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_D = optim.Adam(self.parameters_to_train_D, self.opt.lr)
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(self.model_optimizer_D, 
            self.opt.scheduler_step_size, 0.1)

        self.patch = (1, self.opt.occ_map_size // 2**4, self.opt.occ_map_size // 2**4)

        self.valid = Variable(torch.Tensor(np.ones((self.opt.batch_size, *self.patch))),
                                                     requires_grad=False).float().cuda()
        self.fake  = Variable(torch.Tensor(np.zeros((self.opt.batch_size, *self.patch))),
                                                     requires_grad=False).float().cuda()

        transform = torchvision.transforms.ToTensor()
        labeled_trainset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index,
                                          transform=transform
                                          )
        self.train_loader = DataLoader(labeled_trainset, batch_size= self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers,
                                                  collate_fn=None,pin_memory=True, drop_last=True)

        labeled_valset = LabeledDataset(image_folder=image_folder,
                                        annotation_file=annotation_csv,
                                        scene_index=labeled_scene_index_val,
                                        transform=transform
                                        )
        self.val_loader = DataLoader(labeled_valset, batch_size= 1, shuffle=True, num_workers=self.opt.num_workers,
                                                collate_fn=None,pin_memory=True, drop_last=True)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(labeled_trainset), len(labeled_valset)))


    def train(self):

        for self.epoch in range(self.opt.num_epochs):
            loss = self.run_epoch()
            print("Epoch: %d | Loss: %.4f | Discriminator Loss: %.4f"%(self.epoch, loss["loss"], 
                loss["loss_discr"]))

            if self.epoch % self.opt.log_frequency == 0:
                self.validation()
                self.save_model()

    def run_epoch1(self):
        self.model_optimizer.step()
        # self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0., 0.
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            loss["loss"] += losses["loss"].item()
            loss["loss_discr"] += losses["loss_discr"].item()
        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)
        return loss


    def process_batch(self, inputs, validation=False):
        outputs = {}
        for key, inpt in inputs.items():
            inputs[key] = inpt.to(self.device)
        features = self.models["encoder"](inputs["color"][:,self.opt.pos,:])
        
        outputs["topview"] = self.models["decoder"](features)
        if validation:
            return outputs
        # print(outputs["topview"].shape)
        # print(inputs['dynamic'].shape)
        losses = self.compute_losses(inputs, outputs)
        losses["loss_discr"] = torch.zeros(1)

        return outputs, losses


    def run_epoch(self):
        self.model_optimizer.step()
        self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0.0, 0.0
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            fake_pred = self.models["discriminator"](outputs["topview"])
            #print(inputs["discr"][self.opt.pos].shape)
            real_pred = self.models["discriminator"](inputs["discr"][:,self.opt.pos,:].float())
            #print(fake_pred.shape,real_pred.shape)
            loss_GAN  = self.criterion_d(fake_pred, self.valid)
            loss_D    = self.criterion_d(fake_pred, self.fake)+ self.criterion_d(real_pred, self.valid)
            loss_G    = self.opt.lambda_D * loss_GAN + losses["loss"]

            # Train Discriminator
            if self.epoch > self.opt.discr_train_epoch:
                loss_G.backward(retain_graph=True)
                self.model_optimizer.step()
                self.model_optimizer_D.zero_grad()
                loss_D.backward()
                self.model_optimizer_D.step()
            else:
                losses["loss"].backward()
                self.model_optimizer.step()

            loss["loss"] += losses["loss"].item()
            loss["loss_discr"] += loss_D.item()
        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)
        return loss


    def validation(self):
        iou, mAP = np.array([0., 0.]), np.array([0., 0.])
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            with torch.no_grad():
                outputs = self.process_batch(inputs, True)
            pred = np.squeeze(torch.argmax(outputs["topview"].detach(), 1).cpu().numpy())
            true = np.squeeze(inputs["dynamic"][:,self.opt.pos,:].detach().cpu().numpy())
            iou += mean_IU(pred, true)
            mAP += mean_precision(pred, true)
        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)
        print("Epoch: %d | Validation: mIOU: %.4f mAP: %.4f"%(self.epoch, iou[1], mAP[1]))



    def compute_losses(self, inputs, outputs):
        losses = {}
        losses["loss"] = self.compute_topview_loss(outputs["topview"], inputs["dynamic"][:,self.opt.pos,:],
                self.weight["dynamic"])

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):

        generated_top_view = outputs
        #true_top_view = torch.ones(generated_top_view.size()).cuda()
        #loss = self.weighted_binary_cross_entropy(generated_top_view, true_top_view, torch.Tensor([1, 25]))
        true_top_view = torch.squeeze(true_top_view.long())
        loss = nn.CrossEntropyLoss(weight = torch.Tensor([1., weight]).cuda())
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def save_model(self):
        save_path = os.path.join(self.opt.save_path, self.opt.model_name, "weights_{}".format(self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for key in self.models.keys():
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            #self.optimizer_G.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")




if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
