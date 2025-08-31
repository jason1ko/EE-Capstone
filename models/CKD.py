import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from models.SharedNetwork import SharedNetwork
from misc.utils_python import AverageMeter

class CKD(nn.Module):
    def __init__(self, config):
        super(CKD, self).__init__()

        self.config = config
        self.network = SharedNetwork(num_classes=config.num_classes)

    def forward(self, ocular, face):
        ocular_logit, face_logit, ocular_fea, face_fea = self.network(ocular, face)
        return ocular_logit, face_logit, ocular_fea, face_fea

    def get_train_optimizer(self):
        config = self.config
        optimizer = optim.SGD(self.parameters(), lr=config.lr, momentum=config.momentum,
                              weight_decay=config.weight_decay_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
        return optimizer, scheduler

    def get_pretrain_optimizer(self):
        return None


def train_one_epoch(device, epoch, trainloader, model, optimizer):
    model.train()
    tau = model.tau

    metrics ={
        'loss_class_ocular': AverageMeter(),
        'loss_class_face': AverageMeter(),
        'loss_kd_f2p': AverageMeter(),
        'loss_kd_p2f':AverageMeter()
    }

    for (ocular, face, label) in trainloader:
        batch_size = len(label)
        ocular = ocular.to(device)
        face = face.to(device)
        label = label.to(device)

        ocular_logit, face_logit, _, _ = model(ocular, face)
        num_classes = ocular_logit.shape[-1]
        # y = F.one_hot(label, num_classes=num_classes)

        loss_class_ocular = F.cross_entropy(ocular_logit, label)
        loss_class_face = F.cross_entropy(face_logit, label)
        loss_kd_f2p = tau * tau * F.kl_div(F.log_softmax(ocular_logit / tau, dim=1),
                                           F.softmax(face_logit.detach() / tau, dim=1))
        loss_kd_p2f = tau * tau * F.kl_div(F.log_softmax(face_logit / tau, dim=1),
                                           F.softmax(ocular_logit.detach() / tau, dim=1))

        loss = loss_class_ocular + loss_class_face + loss_kd_f2p + loss_kd_p2f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics['loss_class_ocular'].update(loss_class_ocular.item(), batch_size)
        metrics['loss_class_face'].update(loss_class_face.item(), batch_size)
        metrics['loss_kd_f2p'].update(loss_kd_f2p.item(), batch_size)
        metrics['loss_kd_p2f'].update(loss_kd_p2f.item(), batch_size)

    return metrics

def make_model(config):
    return CKD(config)