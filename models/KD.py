import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch import optim
from tqdm import tqdm


from models.FaceNetwork import FaceNetwork
from models.OcularNetwork import OcularNetwork
from misc.utils_python import AverageMeter

class KD(nn.Module):
    def __init__(self, args):
        super(KD, self).__init__()

        self.tau = args.model_config['tau']

        self.train_config = args.train_config
        self.pretrain_config = args.pretrain_config
        self.model_config = args.model_config

        self.ocular_network = OcularNetwork(num_classes=args.num_classes)
        self.face_network = FaceNetwork(num_classes=args.num_classes)

    def forward(self, ocular, face):
        # face_logit, face_fea = self.face_network(face)
        # ocular_logit, ocular_fea = None, None
        if ocular == None:
            ocular_logit, ocular_fea = None, None
        else:
            ocular_logit, ocular_fea = self.ocular_network(ocular)

        if face == None:
            face_logit, face_fea = None, None
        else:
            face_logit, face_fea = self.face_network(face)

        return ocular_logit, face_logit, ocular_fea, face_fea

    def get_train_optimizer(self):

        lr = self.train_config['lr']
        momentum = self.train_config['momentum']
        weight_decay_rate = self.train_config['weight_decay_rate']
        milestones = self.train_config['milestones']

        train_parameters = self.ocular_network.parameters()
        optimizer = optim.SGD(train_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_pretrain_optimizer(self):
        lr = self.pretrain_config['lr']
        momentum = self.pretrain_config['momentum']
        weight_decay_rate = self.pretrain_config['weight_decay_rate']
        milestones = self.pretrain_config['milestones']

        pretrain_parameters = self.face_network.parameters()
        optimizer = optim.SGD(pretrain_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def freeze_face_network(self):
        for param in self.face_network.parameters():
            param.requires_grad = False

    def unfreeze_face_network(self):
        for param in self.face_network.parameters():
            param.requires_grad = True


def pretrain_one_epoch(device, epoch, trainloader, model, optimizer):
    model.train()

    avg_loss_class_face = AverageMeter()

    pbar = tqdm(trainloader)
    for (_, face, label) in pbar:
        batch_size = len(label)
        face = face.to(device)
        label = label.to(device)

        _, face_logit, _, _ = model(None, face)
        loss_class_face = F.cross_entropy(face_logit, label)

        loss = loss_class_face

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        pbar.set_description(f"loss_class_face: {loss_class_face.item():.5f}")

        avg_loss_class_face.update(loss_class_face.item(), batch_size)

    outputs = {
        'loss_class_face': avg_loss_class_face.avg
    }

    return outputs


def train_one_epoch(device, epoch, trainloader, model, optimizer):
    model.train()
    model.freeze_face_network()
    tau = model.tau

    avg_loss_class_ocular = AverageMeter()
    avg_loss_kd_f2p = AverageMeter()

    pbar = tqdm(trainloader)
    for (ocular, face, label) in pbar:
        batch_size = len(label)
        ocular = ocular.to(device)
        face = face.to(device)
        label = label.to(device)

        ocular_logit, face_logit, _, _ = model(ocular, face)
        # num_classes = ocular_logit.shape[-1]
        # y = F.one_hot(label, num_classes=num_classes)

        loss_class_ocular = F.cross_entropy(ocular_logit, label)

        loss_kd_f2p = tau * tau * F.kl_div(F.log_softmax(ocular_logit / tau, dim=1),
                                           F.softmax(face_logit.detach().clone() / tau, dim=1))


        loss = loss_class_ocular + loss_kd_f2p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        pbar.set_description(f"loss_class_ocular: {loss_class_ocular.item():.5f} "
                             f"loss_kd_f2p: {loss_kd_f2p.item():.5f}")

        avg_loss_class_ocular.update(loss_class_ocular.item(), batch_size)
        avg_loss_kd_f2p.update(loss_kd_f2p.item(), batch_size)


    outputs = {
        'loss_class_ocular': avg_loss_class_ocular.avg,
        'loss_kd_f2p': avg_loss_kd_f2p,
    }

    return outputs

def make_model(config):
    return KD(config)