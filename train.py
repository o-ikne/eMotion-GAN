import argparse
import time
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset.dataset import eMotionGANDataset, load_flow_img_dataset, load_protocol, load_emotions_file

from models.motion_normalizer import MotionNormalizer, MotionDiscriminator, EmotionDiscriminator
from models.motion_warper import MotionWarper
from models.DMUE.eval import load_dmue_model, dmue_inference



def train_epoch(epoch, device, dataloader, motion_normalizer, motion_warper, motion_discriminator, emotion_discriminator):

    motion_normalizer.train()
    discriminator.train()
    emotion_discriminator.train()
    motion_warper.train()

    norm_loss = 0.0
    motion_disc_loss = 0.0
    emotion_disc_loss = 0.0
    warper_loss = 0.0
    
    running_total_loss = 0
    n_iter = epoch*len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, mini_batch in pbar:

        inp_img, tar_img, inp_flow, tar_flow, labels, napex = mini_batch

        ## to device ['cpu' / 'gpu']
        inp_img = inp_img.to(device)
        tar_img = tar_img.to(device)
        inp_flow = inp_flow.to(device)
        tar_flow = tar_flow.to(device)
        labels = labels.to(device)
        
        if any(napex):
            ## image generator
            warp_optimizer.zero_grad()
            real_warped_img = motion_warper(inp_img[napex], tar_flow[napex])
            pred_labels = dmue_inference(dmue_model, real_warped_img, transform, train=True)
            label_loss = categorical_crossentropy(pred_labels, labels[napex].long())
            warper_loss = motion_warper_loss(tar_img[napex], real_warped_img) + label_loss
            warper_loss.backward()
            warp_optimizer.step()

        ## expression discriminator
        emotion_disc_opt.zero_grad()
        disc_real_labels = emotion_discriminator(-.5*tar_flow+.5)
        emotion_disc_loss = categorical_crossentropy(disc_real_labels, labels.long())
        emotion_disc_loss.backward()
        emotion_disc_opt.step()
        
        ## flow generator
        norm_optimizer.zero_grad()
        norm_flow = motion_normalizer(inp_flow)
        disc_output = motion_discriminator(norm_flow)
        disc_gen_labels  = emotion_discriminator(-.5*norm_flow+.5)
        norm_loss, gan_loss = motion_normalizer_loss(norm_flow, tar_flow,
                                            disc_output,
                                            labels, disc_gen_labels)
        if phase > 0:
            if any(napex):
                warped_img = motion_warper(inp_img[napex], norm_flow[napex])
                norm_loss += motion_warper_loss(tar_img[napex], warped_img)
                
        norm_loss.backward()
        norm_optimizer.step()

        ## patch discriminator
        motion_disc_opt.zero_grad()
        fake_logits = motion_discriminator(norm_flow.detach())
        real_logits = motion_discriminator(tar_flow)
        fake_loss = criterion_GAN(fake_logits, torch.zeros_like(fake_logits))
        real_loss = criterion_GAN(real_logits, torch.ones_like(real_logits))
        motion_disc_loss = (real_loss + fake_loss) / 2
        motion_disc_loss.backward()
        motion_disc_opt.step()
        
        running_total_loss += .5 * (norm_loss.item() + warper_loss.item())
        pbar.set_description(f'[TRAIN] EpocH {epoch} => total_loss: %.2f | %.1f' % (running_total_loss / (i + 1), running_total_loss))

    norm_lr_scheduler.step()
    warp_lr_scheduler.step()
    
    return norm_loss, warper_loss, motion_disc_loss, emotion_disc_loss


def evaluate_epoch(epoch, device, dataloader, motion_normalizer, motion_warper, motion_discriminator, emotion_discriminator):

    motion_normalizer.eval()
    motion_discriminator.eval()
    emotion_discriminator.eval()
    motion_warper.eval()

    with torch.no_grad():
        loss_grid_weights = [.5, .5]
        running_total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'EpocH {epoch}')
        loss_array = torch.zeros([len(loss_grid_weights), len(dataloader)], dtype=torch.float32, device=device)

        for i, mini_batch in pbar:

            inp_img, tar_img, inp_flow, tar_flow, labels, _ = mini_batch

            ## to device ['cpu' / 'gpu']
            inp_img = inp_img.to(device)
            tar_img = tar_img.to(device)
            inp_flow = inp_flow.to(device)
            tar_flow = tar_flow.to(device)
            labels = labels.to(device)

            ## warping loss
            norm_flow = motion_normalizer(inp_flow)
            warp_img  = motion_warper(inp_img, gen_flow)
            pred_rgb_labels = dmue_inference(dmue_model, warp_img, transform).to(device)
            rgb_loss = categorical_crossentropy(pred_rgb_labels, labels.long())

            ## normalization loss
            pred_flow_labels = emotion_discriminator(gen_flow)
            flow_loss = categorical_crossentropy(pred_flow_labels, labels.long())

            loss_array[0, i] = rgb_loss
            loss_array[1, i] = flow_loss
            running_total_loss += loss_grid_weights[0] * rgb_loss.item() + loss_grid_weights[1] * flow_loss.item()
            pbar.set_description(f'[VALID] EpocH {epoch} => total loss: %.2f | %.1f' % (running_total_loss / (i + 1), running_total_loss))
    
    mean_loss = torch.mean(loss_array, dim=1)
    mean_loss = loss_grid_weights[0] * mean_loss[0] + loss_grid_weights[1] * mean_loss[1]
    return mean_loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='eMotionGAN trainining')
    ## paths
    parser.add_argument('--exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment')
    parser.add_argument('--data_path', type=str,
                        help='path to data file containing paths to training/validation data.')
    parser.add_argument('--proto_path', type=str,
                        help='protocol path for k-fold evaluation.')
    parser.add_argument('--fold', type=int,
                        help='integer between 0 & k refering to the fold. (k-1) fold for training & 1 fold for evaluation.')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--snapshots', type=str, default='./snapshots')

    ## parameters
    parser.add_argument('--norm_lr', type=float, default=1e-5, help='learning rate for motion normalizer')
    parser.add_argument('--warp_lr', type=float, default=1e-4, help='learning rate for motion warper')
    parser.add_argument('--motion_disc_lr', type=float, default=1e-4, help='learning rate for motion discriminator')
    parser.add_argument('--emotion_disc_lr', type=float, default=1e-4, help='learning rate for emotion discriminator')

    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8, help='number of parallel threads for dataloaders')

    parser.add_argument('--seed', type=int, default=2023,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    ## get trainig & testing folds for eMotionGAN
    protocol = load_protocol(args.proto_path)
    train_fold, valid_fold = train_test_folds(fold, protocol)

    flow_args = {'min_mag': None,
                 'max_mag': 10.0,
                 'shape': (128, 128)}

    # create dataloaders
    train_dataset = eMotionGANDataset(args.data_path, flow_args, train_fold)
    valid_dataset = eMotionGANDataset(args.data_path, flow_args, valid_fold)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=args.n_threads)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.n_threads)


    ## MODELS
    motion_normalizer = MotionNormalizer(in_channels=2, 
                                         out_channels=2, 
                                         dim=64, 
                                         encoder_res=2, 
                                         decoder_res=5, 
                                         n_sampling=4)

    motion_warper  = MotionWarper(motion_channels=2, 
                                  rgb_channels=3,
                                  dim=64, 
                                  encoder_res=4, 
                                  decoder_res=4, 
                                  n_sampling=2)

    motion_discriminator  = MotionDiscriminator(in_channels=2)

    emotion_discriminator = EmotionDiscriminator(in_channels=2, n_classes=6)

    ## load pretrained DMUE model & put it in eval mode
    dmue_model = load_dmue_model()
    dmue_model.eval()

    motion_normalizer = nn.DataParallel(motion_normalizer).to(device)
    motion_discriminator = nn.DataParallel(motion_discriminator).to(device)
    motion_warper = nn.DataParallel(motion_warper).to(device)
    emotion_discriminator = nn.DataParallel(emotion_discriminator).to(device)
    dmue_model = nn.DataParallel(dmue_model).to(device)

    ## Optimizers
    norm_optimizer    = optim.Adam(motion_normalizer.parameters(), lr=args.norm_lr, weight_decay=1e-4)
    warp_optimizer    = optim.Adam(motion_warper.parameters(), lr=args.warp_lr, weight_decay=1e-4)
    motion_disc_opt   = optim.Adam(motion_discriminator.parameters(), lr=args.motion_disc_lr, weight_decay=1e-4)
    emotion_disc_opt  = optim.SGD(emotion_discriminator.parameters(), lr=args.emotion_disc_lr, weight_decay=1e-4)
    norm_lr_scheduler = optim.lr_scheduler.MultiStepLR(norm_optimizer, milestones=[10, 13], gamma=0.5)
    warp_lr_scheduler = optim.lr_scheduler.MultiStepLR(warp_optimizer, milestones=[10, 13], gamma=0.5)

    categorical_crossentropy = nn.CrossEntropyLoss()

    if args.pretrained:
        # reload from pre_trained_model
        motion_normalizer, motion_warper, emotion_discriminator, motion_discriminator,\
        norm_optimizer, warp_optimizer, motion_disc_opt, emotion_disc_opt,\
        norm_lr_scheduler, warp_lr_scheduler,\
        start_epoch, best_val = load_checkpoint(motion_normalizer, motion_warper, emotion_discriminator, motion_discriminator,\
                                                norm_optimizer, warp_optimizer, motion_disc_opt, emotion_disc_opt,\
                                                norm_lr_scheduler, warp_lr_scheduler,\
                                                filename=args.pretrained)

        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))

    else:
        start_epoch = 0
        best_val = float('inf')

        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = args.exp
        if not osp.isdir(os.path.join(args.snapshots, cur_snapshot)):
            os.makedirs(os.path.join(args.snapshots, cur_snapshot))

        with open(os.path.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

    # create summary writer
    save_path = os.path.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    valid_writer = SummaryWriter(os.path.join(save_path, 'valid'))
        
    for epoch in range(start_epoch, args.n_epochs):
        phase = 0
        if epoch >= 5:
            phase = 1

        ## train epoch
        norm_loss, warp_loss, motion_disc_loss, emotion_disc_loss = train_epoch(epoch+1, 
                                                                                device, 
                                                                                train_dataloader,
                                                                                motion_normalizer, 
                                                                                motion_warper, 
                                                                                motion_discriminator,
                                                                                emotion_discriminator)

        print('\t[TRAIN] G_norm: %.3f | G_warp: %.3f | D_patch: %.3f | D_emotion: %.3f' % (norm_loss, warp_loss, motion_disc_loss, emotion_disc_loss))

        train_writer.add_scalar('norm loss', norm_loss, epoch)
        train_writer.add_scalar('warp loss', warp_loss, epoch)
        train_writer.add_scalar('motion disc loss', motion_disc_loss, epoch)
        train_writer.add_scalar('emotion disc loss', emotion_disc_loss, epoch)
        train_writer.add_scalar('norm_learning_rate', norm_lr_scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('warp_learning_rate', warp_lr_scheduler.get_lr()[0], epoch)

        ## eval epoch
        valid_mean_loss = evaluate_epoch(epoch+1, 
                                         device, 
                                         valid_fold,
                                         motion_normalizer, 
                                         motion_warper, 
                                         motion_discriminator, 
                                         emotion_discriminator)
            
        ## save best model
        is_best = val_mean_loss < best_val
        best_val = min(val_mean_loss, best_val)
        print(f"\t[VALID] Mean_loss: %.2f%% | is_best: %s" % (val_mean_loss, is_best))
        valid_writer.add_scalar('valid mean loss', valid_mean_loss, epoch)
            
        save_checkpoint({'epoch': epoch+1,
                         'flow_gen_state_dict': motion_normalizer.state_dict(),
                         'img_gen_state_dict': motion_warper.state_dict(),
                         'emotion_disc_state_dict': emotion_discriminator.state_dict(),
                         'disc_state_dict': motion_discriminator.state_dict(),
                         'flow_gen_optimizer': norm_optimizer.state_dict(),
                         'img_gen_optimizer': warp_optimizer.state_dict(),
                         'disc_optimizer': motion_disc_opt.state_dict(),
                         'emotion_disc_optimizer': emotion_disc_opt.state_dict(),
                         'flow_gen_scheduler': norm_lr_scheduler.state_dict(),
                         'img_gen_scheduler': warp_lr_scheduler.state_dict(),
                         'best_val': best_val},
                          save_path, 'epoch_{}.pth'.format(epoch+1), fold)