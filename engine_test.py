import torch
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity

import gc

@torch.no_grad()
def validate_accuracy(device, model, val_loader):

    model.eval()

    preds_ocular = [[]]*len(val_loader)
    preds_face = [[]]*len(val_loader)
    labels = [[]]*len(val_loader)

    for i, (ocular, face, label) in enumerate(val_loader):
        ocular_logit, face_logit, _, _ = model(ocular.to(device), face.to(device))
        # ocular_logit, face_logit : nn.Linear(512, 1024?? num_classes)

        _, pred_ocular = torch.max(ocular_logit, dim=1)
        _, pred_face = torch.max(face_logit, dim=1)

        preds_ocular[i] = pred_ocular.cpu()
        preds_face[i] = pred_face.cpu()
        labels[i] = label

    preds_ocular = torch.cat(preds_ocular, dim=0)
    preds_face = torch.cat(preds_face, dim=0)
    labels = torch.cat(labels, dim=0)

    acc_ocular = (preds_ocular == labels).mean()
    acc_face = (preds_face == labels).mean()

    return acc_ocular.item(), acc_face.item()



def compute_eer(fpr,tpr): #  """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = np.around(eer, 5) # to make the number neat
    return eer



###############        p2p       ##################


@torch.no_grad()
def compute_periocular_features(device, model, dataloader): # compute_periocular_features
    features = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)

    model.eval()

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader)):
    
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        #_labels = _labels
        # _feas = model.infer_fea(_images)
        _, _, _feas, _ = model(_images_ocular, _images_face)
        _feas = F.normalize(_feas, dim=1) # size is maintained

        #features[i] = _feas.to(device)
        #labels[i] = _labels.to(device)

        features[i] = _feas.cpu()
        labels[i] = _labels.cpu()

        del _images_ocular, _images_face, _feas, _labels
        #gc.collect()
        torch.cuda.empty_cache()

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


@torch.no_grad()
def evaluate_p2p_cmc_cos_sim(device, model, dataloader_gallery, dataloader_probe, top_k=5):
    # model: feature extractor

    model.eval()

    feas_gallery, labels_gallery = compute_periocular_features(device, model, dataloader_gallery)
    feas_gallery_t = feas_gallery.t().to(device) # feas_gallery_t : 512 * len(dataloader_gallery.dataset)
    del feas_gallery

    idxes_max_sims = [[]]*len(dataloader_probe)
    labels = [[]]*len(dataloader_probe)

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader_probe)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        _, _, _feas, _ = model(_images_ocular, _images_face)
        labels[i] = _labels.cpu()
        

        _feas = F.normalize(_feas, dim = 1)
        _sims = torch.mm(_feas, feas_gallery_t).cpu()
        
        _, _idxes_max_sims = torch.topk(_sims, k=top_k, dim=1)

        idxes_max_sims[i] = _idxes_max_sims
        
        del _images_ocular, _images_face, _feas, _labels, _sims, _idxes_max_sims
        #gc.collect()
        torch.cuda.empty_cache()

    idxes_max_sims = torch.cat(idxes_max_sims, dim=0)
    labels = torch.cat(labels, dim=0)

    N_probe = len(labels)

    table_probe = torch.tile(labels_gallery[None, :], (N_probe, 1))
    pred_topk = torch.gather(table_probe, 1, idxes_max_sims)

    labels_topk = torch.tile(labels[:, None], (1, top_k))

    correct_topk = (pred_topk == labels_topk)

    cmc = []
    for k in range(top_k):
        temp = (correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item()
        cmc.append(round(temp, 5))
        #cmc.append((correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item())
    print(cmc[0],'\n')

    return cmc



@torch.no_grad()
def evaluate_p2p_eer(device, model, dataloader):
  model.eval()
  embedding_size = 512 # why 512 ?? bcz the bn5 layer of sharednetwork ends with 512
  batch_size = dataloader.batch_size

  num_data = len(dataloader.dataset)
  num_identity = dataloader.dataset.num_identity
  labels = dataloader.dataset.labels

  embedding_mat = torch.zeros((num_data, embedding_size))

  for i, (_images_ocular, _images_face, _) in tqdm(enumerate(dataloader)):
    _images_ocular = _images_ocular.to(device)
    _images_face = _images_face.to(device)

    num_imgs = _images_ocular.shape[0]
    _, _, _features, _ = model(_images_ocular, _images_face)
    embedding_mat[i*batch_size:i*batch_size+num_imgs, :] = _features.detach().clone()

    del _images_ocular, _images_face
    torch.cuda.empty_cache()

  embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)

  label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
  for i in range(len(labels)):
    label_mat[i, labels[i]] = 1

  score_mat = torch.matmul(embedding_mat, embedding_mat.t()).to(device)
  gen_mat = torch.matmul(label_mat, label_mat.t()).to(device)
  gen_r, gen_c = torch.where(gen_mat == 1)
  imp_r, imp_c = torch.where(gen_mat == 0)

  del embedding_mat, gen_mat #
  torch.cuda.empty_cache()

  gen_score = score_mat[gen_r, gen_c].cpu().numpy()
  imp_score = score_mat[imp_r, imp_c].cpu().numpy()

  del score_mat
  torch.cuda.empty_cache()
  
  y_gen = np.ones(gen_score.shape[0])
  y_imp = np.zeros(imp_score.shape[0])
  
  score = np.concatenate((gen_score, imp_score))
  y = np.concatenate((y_gen, y_imp))

  fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
  eer = compute_eer(fpr_tmp, tpr_tmp)
  
  return eer


####################      f2f      ###################

#my try
@torch.no_grad()
def compute_face_features(device, model, dataloader):
    features = [[]] * len(dataloader) # [[], [], [], ...]
    labels = [[]] * len(dataloader)

    model.eval()

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader)):
      #
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        # _feas = model.infer_fea(_images)
        _, _, _, _feas = model(_images_ocular, _images_face)
        _feas = F.normalize(_feas, dim=1) # dim : which direction to nomalize. size is not changed

        #features[i] = _feas.to(device)
        #labels[i] = _labels.to(device)

        features[i] = _feas.cpu()
        labels[i] = _labels.cpu()

        del _images_ocular, _images_face, _feas
        #gc.collect()
        torch.cuda.empty_cache()

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels




@torch.no_grad()
def evaluate_f2f_cmc_cos_sim(device, model, dataloader_gallery, dataloader_probe, top_k=5):

    model.eval()
    # feas_gallery : len(dataloader_gallery.dataset) * 512

    feas_gallery, labels_gallery = compute_face_features(device, model, dataloader_gallery)
    feas_gallery_t = feas_gallery.t().to(device) # feas_gallery_t : 512 * len(dataloader_gallery.dataset)
    del feas_gallery

    idxes_max_sims = [[]]*len(dataloader_probe)
    labels = [[]]*len(dataloader_probe)

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader_probe)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        _, _, _, _feas = model(_images_ocular, _images_face)
        labels[i] = _labels.cpu()
        #labels[i] = _labels.to(device)

        _feas = F.normalize(_feas, dim = 1)
        _sims = torch.mm(_feas, feas_gallery_t).cpu()
        
        _, _idxes_max_sims = torch.topk(_sims, k=top_k, dim=1)

        idxes_max_sims[i] = _idxes_max_sims
        #idxes_max_sims[i] = _idxes_max_sims.to(device)

        #del _images_ocular, _images_face
        del _images_ocular, _images_face, _feas, _labels, _sims, _idxes_max_sims
        #gc.collect()
        torch.cuda.empty_cache()

    idxes_max_sims = torch.cat(idxes_max_sims, dim=0)
    labels = torch.cat(labels, dim=0)

    N_probe = len(labels)

    table_probe = torch.tile(labels_gallery[None, :], (N_probe, 1))
    pred_topk = torch.gather(table_probe, 1, idxes_max_sims)

    labels_topk = torch.tile(labels[:, None], (1, top_k))

    correct_topk = (pred_topk == labels_topk)

    cmc = []
    for k in range(top_k):
        temp = (correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item()
        cmc.append(round(temp, 5))
        #cmc.append((correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item())
    print(cmc[0],'\n')

    return cmc




@torch.no_grad()
def evaluate_f2f_eer(device, model, dataloader):
  model.eval()
  embedding_size = 512 # why 512 ?? bcz the bn5 layer of sharednetwork ends with 512
  batch_size = dataloader.batch_size

  num_data = len(dataloader.dataset)
  num_identity = dataloader.dataset.num_identity
  labels = dataloader.dataset.labels

  embedding_mat = torch.zeros((num_data, embedding_size))

  for i, (_images_ocular, _images_face, _) in tqdm(enumerate(dataloader)):
    _images_ocular = _images_ocular.to(device)
    _images_face = _images_face.to(device)

    num_imgs = _images_ocular.shape[0]
    _, _, _, _features = model(_images_ocular, _images_face)
    embedding_mat[i*batch_size:i*batch_size+num_imgs, :] = _features.detach().clone()

    del _images_ocular, _images_face
    torch.cuda.empty_cache()

  embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)

  label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
  for i in range(len(labels)):
    label_mat[i, labels[i]] = 1

  score_mat = torch.matmul(embedding_mat, embedding_mat.t()).to(device)
  gen_mat = torch.matmul(label_mat, label_mat.t()).to(device)
  gen_r, gen_c = torch.where(gen_mat == 1)
  imp_r, imp_c = torch.where(gen_mat == 0)

  del embedding_mat, gen_mat #
  torch.cuda.empty_cache()

  gen_score = score_mat[gen_r, gen_c].cpu().numpy()
  imp_score = score_mat[imp_r, imp_c].cpu().numpy()

  del score_mat
  torch.cuda.empty_cache()
  
  y_gen = np.ones(gen_score.shape[0])
  y_imp = np.zeros(imp_score.shape[0])
  
  score = np.concatenate((gen_score, imp_score))
  y = np.concatenate((y_gen, y_imp))

  fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
  eer = compute_eer(fpr_tmp, tpr_tmp)
  
  return eer


###########################      p2f    ########################


@torch.no_grad()
def compute_periocular_logits(device, model, dataloader):
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)

    model.eval()

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        _logits, _, _, _ = model(_images_ocular, _images_face)
        _logits = F.normalize(_logits, dim=1) # size is maintained
        #_logits = F.softmax(_logits, dim=1) # size is maintained

        logits[i] = _logits.cpu()
        labels[i] = _labels.cpu()

        #logits[i] = _logits.to(device)
        #labels[i] = _labels.to(device)

        del _images_ocular, _images_face, _logits, _labels
        #gc.collect()
        torch.cuda.empty_cache()

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    return logits, labels


@torch.no_grad()
def evaluate_p2f_cmc_cos_sim(device, model, dataloader_gallery, dataloader_probe, top_k=5):

    model.eval()

    logits_g, labels_gallery = compute_periocular_logits(device, model, dataloader_gallery)
    logits_g_t = logits_g.t().to(device) # feas_gallery_t : 512 * len(dataloader_gallery.dataset)
    del logits_g

    idxes_max_sims = [[]]*len(dataloader_probe)
    labels = [[]]*len(dataloader_probe)

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader_probe)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        _, _logits, _, _ = model(_images_ocular, _images_face)
        
        labels[i] = _labels.cpu()
        #labels[i] = _labels.to(device)

        _logits = F.normalize(_logits, dim=1)
        _sims = torch.mm(_logits, logits_g_t).cpu()
        #_sims = torch.mm(_logits, logits_g_t).to(device)
        _, _idxes_max_sims = torch.topk(_sims, k=top_k, dim=1)

        idxes_max_sims[i] = _idxes_max_sims

        #del _images_ocular, _images_face
        del _images_ocular, _images_face, _logits, _labels, _sims, _idxes_max_sims
        #gc.collect()
        torch.cuda.empty_cache()

    idxes_max_sims = torch.cat(idxes_max_sims, dim=0)
    labels = torch.cat(labels, dim=0)

    N_probe = len(labels)

    table_probe = torch.tile(labels_gallery[None, :], (N_probe, 1))
    pred_topk = torch.gather(table_probe, 1, idxes_max_sims)

    labels_topk = torch.tile(labels[:, None], (1, top_k))

    correct_topk = (pred_topk == labels_topk)

    cmc = []
    for k in range(top_k):
        temp = (correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item()
        cmc.append(round(temp, 5))
        #cmc.append((correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item())
    
    print(cmc[0],'\n')

    return cmc


# not finished
@torch.no_grad()
def calculate_kl_div(_logits_probe, logits_gallery, T, device):
    
    _sims_kldiv = torch.zeros(_logits_probe.shape[0], logits_gallery.shape[0]).to(device)
    
    for j in range(_logits_probe.shape[0]):
        for k in range(logits_gallery.shape[0]):
            _sims_kldiv[j][k] = F.kl_div(logits_gallery[k]/T, _logits_probe[j]/T, reduction = 'batchmean')
    
    return _sims_kldiv



@torch.no_grad()
def evaluate_p2f_cmc_kl_div(device, model, dataloader_gallery, dataloader_probe,T=1, top_k=5,):

    model.eval()

    logits_g, labels_gallery = compute_periocular_logits(device, model, dataloader_gallery)
    logits_g = logits_g.to(device)

    idxes_max_sims = [[]]*len(dataloader_probe)
    labels = [[]]*len(dataloader_probe)

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader_probe)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        _, _logits, _, _ = model(_images_ocular, _images_face)
        labels[i] = _labels.cpu()
        #_logits = F.softmax(_logits, dim = 1)
        
        _temp_logits = _logits.clone()
        _temp_logits[_logits <= 0] = 1.

        _sims = (_logits[:, None,:]/T * (torch.log(_temp_logits[:, None,:]/T) - logits_g[None, :, :]/T)).mean(dim=-1).cpu()
        _, _idxes_max_sims = torch.topk(-_sims, k=top_k, dim=1)

        idxes_max_sims[i] = _idxes_max_sims

        #del _images_ocular, _images_face
        del _images_ocular, _images_face, _logits, _labels, _sims, _idxes_max_sims
        #gc.collect()
        torch.cuda.empty_cache()

    idxes_max_sims = torch.cat(idxes_max_sims, dim=0)
    labels = torch.cat(labels, dim=0)

    N_probe = len(labels)

    table_probe = torch.tile(labels_gallery[None, :], (N_probe, 1))
    pred_topk = torch.gather(table_probe, 1, idxes_max_sims)

    labels_topk = torch.tile(labels[:, None], (1, top_k))

    correct_topk = (pred_topk == labels_topk)

    cmc = []
    for k in range(top_k):
        temp = (correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item()
        cmc.append(round(temp, 5))
        #cmc.append((correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item())
    
    print(cmc[0],'\n')

    return cmc



@torch.no_grad()
def evaluate_p2f_eer_cos_sim(device, model, dataloader):
    model.eval()
    embedding_size = 1054
    batch_size = dataloader.batch_size # 32
    
    num_data = len(dataloader.dataset)
    num_identity = dataloader.dataset.num_identity
    labels = dataloader.dataset.labels
    
    embedding_mat_peri = torch.zeros((num_data, embedding_size))# for peri
    embedding_mat_face = torch.zeros((num_data, embedding_size))
    
    for i, (_images_ocular, _images_face, _) in tqdm(enumerate(dataloader)):# for peri
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        num_imgs = _images_ocular.shape[0] # batch size = 32
        _logits_ocular, _logits_face, _, _ = model(_images_ocular, _images_face)
        
        embedding_mat_peri[i*batch_size:i*batch_size+num_imgs, :] = _logits_ocular.detach().clone()
        embedding_mat_face[i*batch_size:i*batch_size+num_imgs, :] = _logits_face.detach().clone()
        
        del _images_ocular, _images_face
        torch.cuda.empty_cache()
        
    embedding_mat_peri /= torch.norm(embedding_mat_peri, p=2, dim=1, keepdim=True) #normalization by row
    embedding_mat_face /= torch.norm(embedding_mat_face, p=2, dim=1, keepdim=True)

    label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
    
    for i in range(len(labels)):
        label_mat[i, labels[i]] = 1

    score_mat = torch.matmul(embedding_mat_peri, embedding_mat_face.t()).to(device) # peri * face^T, cosine sim.
    gen_mat = torch.matmul(label_mat, label_mat.t()).to(device)
    gen_r, gen_c = torch.where(gen_mat == 1)
    imp_r, imp_c = torch.where(gen_mat == 0)

    del embedding_mat_peri, embedding_mat_face, gen_mat #
    torch.cuda.empty_cache()

    gen_score = score_mat[gen_r, gen_c].cpu().numpy()
    imp_score = score_mat[imp_r, imp_c].cpu().numpy()

    del score_mat
    torch.cuda.empty_cache()
  
    y_gen = np.ones(gen_score.shape[0])
    y_imp = np.zeros(imp_score.shape[0])
  
    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((y_gen, y_imp))

    fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
    eer = compute_eer(fpr_tmp, tpr_tmp)
  
    return eer




@torch.no_grad()
def evaluate_p2f_eer_kl_div(device, model, dataloader, T = 1):
    model.eval()
    embedding_size = 1054
    batch_size = dataloader.batch_size # 32
    
    num_data = len(dataloader.dataset)
    num_identity = dataloader.dataset.num_identity
    labels = dataloader.dataset.labels
    
    embedding_mat_peri = torch.zeros((num_data, embedding_size))# for peri
    embedding_mat_face = torch.zeros((num_data, embedding_size))
    
    label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
    
    
    for i, (_images_ocular, _images_face, _) in tqdm(enumerate(dataloader)):# for peri
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        num_imgs = _images_ocular.shape[0] # batch size = 32
        _logits_ocular, _logits_face, _, _ = model(_images_ocular, _images_face)
        
        embedding_mat_peri[i*batch_size:i*batch_size+num_imgs, :] = _logits_ocular.detach().clone()
        embedding_mat_face[i*batch_size:i*batch_size+num_imgs, :] = _logits_face.detach().clone()
        
        del _images_ocular, _images_face
        torch.cuda.empty_cache()
    
    label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
    
    for i in range(len(labels)):
        label_mat[i, labels[i]] = 1

    
    embedding_mat_peri = embedding_mat_peri.cpu()
    embedding_mat_face = embedding_mat_face.cpu()
    
    temp_peri = embedding_mat_peri.clone()
    temp_peri[embedding_mat_peri <= 0] = 1.0

    score_mat = (embedding_mat_peri[:, None,:]/T * (torch.log(temp_peri[:, None,:]/T) - \
                                                    embedding_mat_face[None, :, :]/T)).mean(dim=-1).cpu()
    score_mat *= -1.0
    
    gen_mat = torch.matmul(label_mat, label_mat.t()).to(device)
    gen_r, gen_c = torch.where(gen_mat == 1)
    imp_r, imp_c = torch.where(gen_mat == 0)

    del embedding_mat_peri, embedding_mat_face, gen_mat #
    torch.cuda.empty_cache()

    gen_score = score_mat[gen_r, gen_c].cpu().numpy()
    imp_score = score_mat[imp_r, imp_c].cpu().numpy()

    del score_mat
    torch.cuda.empty_cache()
  
    y_gen = np.ones(gen_score.shape[0])
    y_imp = np.zeros(imp_score.shape[0])
  
    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((y_gen, y_imp))

    fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
    eer = compute_eer(fpr_tmp, tpr_tmp)
  
    return eer


###########################      f2p     ########################

@torch.no_grad()
def compute_face_logits(device, model, dataloader):
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)

    model.eval()

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        _, _logits, _, _ = model(_images_ocular, _images_face)
        _logits = F.normalize(_logits, dim=1) # size is maintained
        #_logits = F.softmax(_logits, dim=1) # size is maintained

        logits[i] = _logits.cpu()
        labels[i] = _labels.cpu()

        del _images_ocular, _images_face, _logits, _labels
        #gc.collect()
        torch.cuda.empty_cache()

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    return logits, labels


@torch.no_grad()
def evaluate_f2p_cmc_cos_sim(device, model, dataloader_gallery, dataloader_probe, top_k=5):

    model.eval()

    logits_g, labels_gallery = compute_face_logits(device, model, dataloader_gallery)
    logits_g_t = logits_g.t().to(device)
    del logits_g

    idxes_max_sims = [[]]*len(dataloader_probe)
    labels = [[]]*len(dataloader_probe)

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader_probe)):#peri
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        _logits, _, _, _ = model(_images_ocular, _images_face)
        
        labels[i] = _labels.cpu()
        #labels[i] = _labels.to(device)

        _logits = F.normalize(_logits, dim=1)
        _sims = torch.mm(_logits, logits_g_t).cpu()
        #_sims = torch.mm(_logits, logits_g_t).to(device)
        _, _idxes_max_sims = torch.topk(_sims, k=top_k, dim=1)

        idxes_max_sims[i] = _idxes_max_sims

        #del _images_ocular, _images_face
        del _images_ocular, _images_face, _logits, _labels, _sims, _idxes_max_sims
        #gc.collect()
        torch.cuda.empty_cache()

    idxes_max_sims = torch.cat(idxes_max_sims, dim=0)
    labels = torch.cat(labels, dim=0)

    N_probe = len(labels)

    table_probe = torch.tile(labels_gallery[None, :], (N_probe, 1))
    pred_topk = torch.gather(table_probe, 1, idxes_max_sims)

    labels_topk = torch.tile(labels[:, None], (1, top_k))

    correct_topk = (pred_topk == labels_topk)

    cmc = []
    for k in range(top_k):
        temp = (correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item()
        cmc.append(round(temp, 5))
        #cmc.append((correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item())
    
    print(cmc[0],'\n')

    return cmc




@torch.no_grad()
def evaluate_f2p_cmc_kl_div(device, model, dataloader_gallery, dataloader_probe,T=1, top_k=5,):

    model.eval()

    logits_g, labels_gallery = compute_face_logits(device, model, dataloader_gallery)
    logits_g = logits_g.to(device)

    idxes_max_sims = [[]]*len(dataloader_probe)
    labels = [[]]*len(dataloader_probe)

    for i, (_images_ocular, _images_face, _labels) in tqdm(enumerate(dataloader_probe)):
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        _logits, _, _, _ = model(_images_ocular, _images_face)
        labels[i] = _labels.cpu()
        #_logits = F.softmax(_logits, dim = 1)
        
        _temp_logits = _logits.clone()
        _temp_logits[_logits <= 0] = 1.

        _sims = (_logits[:, None,:]/T * (torch.log(_temp_logits[:, None,:]/T) - logits_g[None, :, :]/T)).mean(dim=-1).cpu()
        _, _idxes_max_sims = torch.topk(-_sims, k=top_k, dim=1)

        idxes_max_sims[i] = _idxes_max_sims

        #del _images_ocular, _images_face
        del _images_ocular, _images_face, _logits, _labels, _sims, _idxes_max_sims
        #gc.collect()
        torch.cuda.empty_cache()

    idxes_max_sims = torch.cat(idxes_max_sims, dim=0)
    labels = torch.cat(labels, dim=0)

    N_probe = len(labels)

    table_probe = torch.tile(labels_gallery[None, :], (N_probe, 1))
    pred_topk = torch.gather(table_probe, 1, idxes_max_sims)

    labels_topk = torch.tile(labels[:, None], (1, top_k))

    correct_topk = (pred_topk == labels_topk)

    cmc = []
    for k in range(top_k):
        temp = (correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item()
        cmc.append(round(temp, 5))
        #cmc.append((correct_topk[:, :k + 1].sum(dim=1) >= 1).float().mean().item())
    
    print(cmc[0],'\n')

    return cmc




@torch.no_grad()
def evaluate_f2p_eer_cos_sim(device, model, dataloader):
    model.eval()
    embedding_size = 1054
    batch_size = dataloader.batch_size # 32
    
    num_data = len(dataloader.dataset)
    num_identity = dataloader.dataset.num_identity
    labels = dataloader.dataset.labels
    
    embedding_mat_face = torch.zeros((num_data, embedding_size))
    embedding_mat_peri = torch.zeros((num_data, embedding_size))# for peri
    
    
    for i, (_images_ocular, _images_face, _) in tqdm(enumerate(dataloader)):# for peri
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        num_imgs = _images_ocular.shape[0] # batch size = 32
        _logits_ocular, _logits_face, _, _ = model(_images_ocular, _images_face)
        
        embedding_mat_face[i*batch_size:i*batch_size+num_imgs, :] = _logits_face.detach().clone()
        embedding_mat_peri[i*batch_size:i*batch_size+num_imgs, :] = _logits_ocular.detach().clone()
                
        del _images_ocular, _images_face
        torch.cuda.empty_cache()
        
    embedding_mat_face /= torch.norm(embedding_mat_face, p=2, dim=1, keepdim=True)
    embedding_mat_peri /= torch.norm(embedding_mat_peri, p=2, dim=1, keepdim=True) #normalization by row

    label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
    
    for i in range(len(labels)):
        label_mat[i, labels[i]] = 1

    score_mat = torch.matmul(embedding_mat_face, embedding_mat_peri.t()).to(device) # peri * face^T, cosine sim.
    gen_mat = torch.matmul(label_mat, label_mat.t()).to(device)
    gen_r, gen_c = torch.where(gen_mat == 1)
    imp_r, imp_c = torch.where(gen_mat == 0)

    del embedding_mat_peri, embedding_mat_face, gen_mat #
    torch.cuda.empty_cache()

    gen_score = score_mat[gen_r, gen_c].cpu().numpy()
    imp_score = score_mat[imp_r, imp_c].cpu().numpy()

    del score_mat
    torch.cuda.empty_cache()
  
    y_gen = np.ones(gen_score.shape[0])
    y_imp = np.zeros(imp_score.shape[0])
  
    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((y_gen, y_imp))

    fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
    eer = compute_eer(fpr_tmp, tpr_tmp)
  
    return eer




@torch.no_grad()
def evaluate_f2p_eer_kl_div(device, model, dataloader, T = 1):
    model.eval()
    embedding_size = 1054
    batch_size = dataloader.batch_size # 32
    
    num_data = len(dataloader.dataset)
    num_identity = dataloader.dataset.num_identity
    labels = dataloader.dataset.labels
    
    embedding_mat_face = torch.zeros((num_data, embedding_size))
    embedding_mat_peri = torch.zeros((num_data, embedding_size))# for peri
    
    label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
    
    
    for i, (_images_ocular, _images_face, _) in tqdm(enumerate(dataloader)):# for peri
        _images_ocular = _images_ocular.to(device)
        _images_face = _images_face.to(device)
        
        num_imgs = _images_ocular.shape[0] # batch size = 32
        _logits_ocular, _logits_face, _, _ = model(_images_ocular, _images_face)
        
        embedding_mat_face[i*batch_size:i*batch_size+num_imgs, :] = _logits_face.detach().clone()
        embedding_mat_peri[i*batch_size:i*batch_size+num_imgs, :] = _logits_ocular.detach().clone()
        
        del _images_ocular, _images_face
        torch.cuda.empty_cache()
    
    label_mat = torch.zeros((len(labels), num_identity)) # 4 * 1
    
    for i in range(len(labels)):
        label_mat[i, labels[i]] = 1

    embedding_mat_face = embedding_mat_face.cpu()
    embedding_mat_peri = embedding_mat_peri.cpu()
        
    temp_face = embedding_mat_face.clone()
    temp_face[embedding_mat_face <= 0] = 1.0

    score_mat = (embedding_mat_face[:, None,:]/T * (torch.log(temp_face[:, None,:]/T) - \
                                                    embedding_mat_peri[None, :, :]/T)).mean(dim=-1).cpu()
    score_mat *= -1.0
    
    gen_mat = torch.matmul(label_mat, label_mat.t()).to(device)
    gen_r, gen_c = torch.where(gen_mat == 1)
    imp_r, imp_c = torch.where(gen_mat == 0)

    del embedding_mat_peri, embedding_mat_face, gen_mat #
    torch.cuda.empty_cache()

    gen_score = score_mat[gen_r, gen_c].cpu().numpy()
    imp_score = score_mat[imp_r, imp_c].cpu().numpy()

    del score_mat
    torch.cuda.empty_cache()
  
    y_gen = np.ones(gen_score.shape[0])
    y_imp = np.zeros(imp_score.shape[0])
  
    score = np.concatenate((gen_score, imp_score))
    y = np.concatenate((y_gen, y_imp))

    fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
    eer = compute_eer(fpr_tmp, tpr_tmp)
  
    return eer