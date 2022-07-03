import argparse
import logging
import os
import datetime
import random
from copy import deepcopy

import torch
import torch.optim as Optim

from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from dataset import ShapeNet
from models import PCN, VN_PCN
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, emd_loss
from visualization import plot_pcd_one_view
from utils.experiments import get_num_params


log = logging.getLogger("train")
log_dataset = logging.getLogger("dataset")

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def prepare_tf_writters(config):
    # prepare logger directory
    visual_dir = os.path.join(config.exp_dir, 'visualizations')
    model_dir = os.path.join(config.exp_dir, 'models')
    optim_dir = os.path.join(config.exp_dir, 'optimizer')
    train_writer = SummaryWriter(os.path.join(config.exp_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(config.exp_dir, 'val'))

    return visual_dir, model_dir, optim_dir, train_writer, val_writer


def train(config, args):
    torch.backends.cudnn.benchmark = True

    visual_dir, model_dir, optim_dir, train_writer, val_writer = prepare_tf_writters(config)

    log_dataset.info('Loading Data...')

    train_dataset = ShapeNet('data/PCN', 'train', config.category)
    val_dataset = ShapeNet('data/PCN', 'valid', config.category)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    log_dataset.info("Dataset loaded!")

    # model
    if config.VN:
        model = VN_PCN(num_dense=16384, latent_dim=1024, grid_size=4, only_coarse=config.only_coarse).to(config.device)
    else:
        model = PCN(num_dense=16384, latent_dim=1024, grid_size=4, only_coarse=config.only_coarse).to(config.device)

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    start_epoch = 0
    if args.resume:
        exp_dir = config.exp_dir
        model_path = os.path.join(exp_dir, 'models/model_last.pth')
        optim_path = os.path.join(exp_dir, 'optimizer/optim_last.pth')
        if os.path.exists(model_path) and os.path.exists(optim_path):
            log.info(f'Resume training from experiment: {args.name}')
            model.load_state_dict(torch.load(model_path))
            optim_dict = torch.load(optim_path)
            optimizer.load_state_dict(optim_dict['optim_state_dict'])
            start_epoch = optim_dict['epoch']
        else:
            log.info(f'Tried to resume training from experiment: {args.exp_name}, however, model.pth or optim.pth not existant. Train from start')

    scheduler = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    step = len(train_dataloader) // config.log_frequency
    n_batches = len(train_dataloader)
    
    model_params_dict = get_num_params(model)
    log.info(f"Model coarse part params {model_params_dict['coarse']}")
    log.info(f"Model dense part params {model_params_dict['dense']}")
    # load pretrained model and optimizer

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    for epoch in range(start_epoch, config.max_epochs + 1):
        # hyperparameter alpha
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif epoch < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        # training
        model.train()
        for i, (p, c) in enumerate(train_dataloader):
            p, c = p.to(config.device), c.to(config.device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p)
            
            # loss function
            if config.coarse_loss == 'cd':
                loss1 = cd_loss_L1(coarse_pred, c)
            elif config.coarse_loss == 'emd':
                coarse_c = c[:, :1024, :]
                loss1 = emd_loss(coarse_pred, coarse_c)
            else:
                raise ValueError('Not implemented loss {}'.format(config.coarse_loss))
            
            
            if config.only_coarse:
                loss2 = torch.zeros(1)
                loss = loss1
            else:
                loss2 = cd_loss_L1(dense_pred, c)
                loss = loss1 + alpha * loss2  

            # back propagation
            loss.backward()
            optimizer.step()

            if (i + 1) % step == 0:
                log.info("Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, dense l1 cd = {:.6f}, total loss = {:.6f}, lr = {:.6f}"
                    .format(epoch, config.max_epochs, i + 1, len(train_dataloader), loss1.item() * 1e3, loss2.item() * 1e3, loss.item() * 1e3, scheduler.get_last_lr()[0]))
            
            train_step = epoch * n_batches + i
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
        
        scheduler.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, (p, c) in enumerate(val_dataloader):
                p, c = p.to(config.device), c.to(config.device)
                coarse_pred, dense_pred = model(p)
                if config.only_coarse:
                    total_cd_l1 += l1_cd(coarse_pred, c).item()
                else:
                    total_cd_l1 += l1_cd(dense_pred, c).item()

                # save into image
                if rand_iter == i:
                    index = random.randint(0, coarse_pred.shape[0] - 1)
                    if config.only_coarse:
                        plot_pcd_one_view(os.path.join(visual_dir, 'epoch_{:03d}.png'.format(epoch)),
                                        [p[index].detach().cpu().numpy(), coarse_pred[index].detach().cpu().numpy(), c[index].detach().cpu().numpy()],
                                        ['Input', 'Coarse', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    else:
                        plot_pcd_one_view(os.path.join(visual_dir, 'epoch_{:03d}.png'.format(epoch)),
                                        [p[index].detach().cpu().numpy(), coarse_pred[index].detach().cpu().numpy(), dense_pred[index].detach().cpu().numpy(), c[index].detach().cpu().numpy()],
                                        ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            
            total_cd_l1 /= len(val_dataset)
            val_writer.add_scalar('l1_cd', total_cd_l1, epoch)
            val_step += 1

            log.info("Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, config.max_epochs, total_cd_l1 * 1e3))
        
        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(model_dir, 'model_best.pth'))
            state_dict = deepcopy(optimizer.state_dict())

            torch.save(
                {"epoch": epoch, "optim_state_dict": state_dict},
                os.path.join(optim_dir, 'optim_best.pth')
            )
        
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_last.pth'))
        state_dict = deepcopy(optimizer.state_dict())

        torch.save(
            {"epoch": epoch, "optim_state_dict": state_dict},
            os.path.join(optim_dir, 'optim_last.pth')
        )
            
    log.info('Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCN')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--only_coarse', action='store_true', help='Train on coarse prediction only')
    parser.add_argument('--VN', action='store_true', help='use VN network')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--resume', action='store_true', help='Resume training specified by the exp_name')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    args = parser.parse_args()
    
    train(args)
