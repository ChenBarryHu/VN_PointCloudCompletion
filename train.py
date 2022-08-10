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
from models.dgcnn import DGCNN, DGCNN_fps
from visualization import plot_pcd_one_view
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from utils.experiments import get_num_params, get_num_params_total
from utils.loss import calc_dcd
from models.model import PCNNet


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


    model = PCNNet(config, enc_type=config.enc_type, dec_type=config.dec_type, resume=args.resume)
    if config.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
    if config.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    start_epoch = 0
    if args.resume:
        exp_dir = config.exp_dir
        model_path = os.path.join(exp_dir, 'models/model_last.pth')
        optim_path = os.path.join(exp_dir, 'optimizer/optim_last.pth')
        if os.path.exists(model_path) and os.path.exists(optim_path):
            model.load_state_dict(torch.load(model_path))
            optim_dict = torch.load(optim_path)
            optimizer.load_state_dict(optim_dict['optim_state_dict'])
            start_epoch = optim_dict['epoch'] + 1
            best_cd_l1 = optim_dict['best_metrics']
            best_epoch_l1 = optim_dict['best_epoch']
            log.info(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_cd_l1 * 1e3):s})')
        else:
            log.info(f'Tried to resume training from experiment: {args.exp_name}, however, model.pth or optim.pth not existant. Train from start')
            best_cd_l1 = 1e8
            best_epoch_l1 = -1
    else:
        best_cd_l1 = 1e8
        best_epoch_l1 = -1
        log.info(f'Start a brand new experiment: {config.run_name}')

    scheduler = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    step = len(train_dataloader) // config.log_frequency
    n_batches = len(train_dataloader)
    
    model_total_params = get_num_params_total(model)
    log.info(f"Model total params: {model_total_params}")
    log.info(f"Producing coarse only: {config.only_coarse}")
    log.info(f"Producing num of coarse points: {config.num_coarse}")
    # model_params_dict = get_num_params(model)
    # log.info(f"Model coarse part params {model_params_dict['coarse']}")
    # log.info(f"Model dense part params {model_params_dict['dense']}")
    # load pretrained model and optimizer

    # training
    train_step, val_step = start_epoch * n_batches, 0
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
        train_cd_l1 = {
            "coarse" : 0.0,
            "dense" : 0.0,
            "total" : 0.0
        }
        for i, (p, c) in enumerate(train_dataloader):
            p, c = p.to(config.device), c.to(config.device)
            # print(f"Input pointcloud shape: {p.shape}\n")
            trot = None
            if config.rotation == 'z':
                trot = RotateAxisAngle(angle=torch.rand(p.shape[0])*360, axis="Z", degrees=True).to(config.device)
            elif  config.rotation == 'so3':
                trot = Rotate(R=random_rotations(p.shape[0])).to(config.device)

            if trot is not None:
                p = trot.transform_points(p)
                c = trot.transform_points(c)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p, trot)
            
            # loss function
            if config.coarse_loss == 'cd':
                loss1 = cd_loss_L1(coarse_pred, c)
            elif config.coarse_loss == 'emd':
                coarse_c = c[:, :1024, :]
                loss1 = emd_loss(coarse_pred, coarse_c)
            elif config.coarse_loss == 'dcd':
                t_alpha = config.dcd_opts["alpha"]
                n_lambda = config.dcd_opts["lambda"]
                loss1, _, _ = calc_dcd(coarse_pred, c, alpha=t_alpha, n_lambda=n_lambda)
                loss1 = loss1.mean()
            else:
                raise ValueError('Not implemented loss {}'.format(config.coarse_loss))
            
            
            if config.only_coarse:
                loss2 = torch.zeros(1)
                loss = loss1
            else:
                loss2 = cd_loss_L1(dense_pred, c)
                loss = loss2
                # loss2 = cd_loss_L1(dense_pred, c)
                # loss = loss1 + alpha * loss2
                train_cd_l1["dense"] += loss2.item()

            # back propagation
            loss.backward()
            optimizer.step()

            train_cd_l1["total"] += loss.item()
            train_cd_l1["coarse"] += loss1.item()
            

            if (i + 1) % step == 0:
                log.info("Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, dense loss = {:.6f}, total loss = {:.6f}, lr = {:.6f}"
                    .format(epoch, config.max_epochs, i + 1, len(train_dataloader), loss1.item() * 1e3, loss2.item() * 1e3, loss.item() * 1e3, scheduler.get_last_lr()[0]))
            
            train_step = epoch * n_batches + i
            train_writer.add_scalar('Loss/Batch/Coarse', loss1.item(), train_step)
            train_writer.add_scalar('Loss/Batch/Dense', loss2.item(), train_step)
            train_writer.add_scalar('Loss/Batch/Total', loss.item(), train_step)
        
        scheduler.step()
        train_loss_epoch = train_cd_l1["total"] / len(train_dataloader)
        train_coarse_loss_epoch = train_cd_l1["coarse"] / len(train_dataloader)
        train_dense_loss_epoch = train_cd_l1["dense"] / len(train_dataloader)
        log.info("Training Epoch [{:03d}/{:03d}]: Coarse Loss = {:.6f}, Dense Loss = {:.6f}, Total Loss = {:.6f}".format(
            epoch, config.max_epochs, train_coarse_loss_epoch * 1e3, train_dense_loss_epoch * 1e3, train_loss_epoch * 1e3))
        train_writer.add_scalar('Loss/Epoch/Coarse', train_coarse_loss_epoch * 1e3, epoch)
        train_writer.add_scalar('Loss/Epoch/Dense', train_dense_loss_epoch * 1e3, epoch)
        train_writer.add_scalar('Loss/Epoch/Total', train_loss_epoch * 1e3, epoch)

        # evaluation
        model.eval()
        val_loss = {
            "coarse" : 0.0,
            "dense" : 0.0,
            "total" : 0.0,
        }
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, (p, c) in enumerate(val_dataloader):
                p, c = p.to(config.device), c.to(config.device)
                trot = None
                if config.val_rotation == 'z':
                    trot = RotateAxisAngle(angle=torch.rand(p.shape[0])*360, axis="Z", degrees=True).to(config.device)
                elif  config.val_rotation == 'so3':
                    trot = Rotate(R=random_rotations(p.shape[0])).to(config.device)

                if trot is not None:
                    p = trot.transform_points(p)
                    c = trot.transform_points(c)

                coarse_pred, dense_pred = model(p, trot)
                val_loss["coarse"] += l1_cd(coarse_pred, c).item()
                if config.only_coarse:
                    val_loss["total"] = val_loss["coarse"]
                else:
                    val_loss["dense"] += l1_cd(dense_pred, c).item()
                    val_loss["total"] = val_loss["coarse"] + val_loss["dense"]

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
            
            val_loss["coarse"] /= len(val_dataset)
            val_loss["dense"] /= len(val_dataset)
            val_loss["total"] /= len(val_dataset)

            val_writer.add_scalar('Loss/Epoch/Coarse', val_loss["coarse"] * 1e3, epoch)
            val_writer.add_scalar('Loss/Epoch/Dense', val_loss["dense"] * 1e3, epoch)
            val_writer.add_scalar('Loss/Epoch/Total', val_loss["total"] * 1e3, epoch)
            val_step += 1

            log.info("Validate Epoch [{:03d}/{:03d}]: Coarse Loss = {:.6f}, Dense Loss = {:.6f}, Total Loss = {:.6f}".format(
            epoch, config.max_epochs, val_loss["coarse"] * 1e3, val_loss["dense"] * 1e3, val_loss["dense"] * 1e3))
        
        if val_loss["total"] < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(model_dir, 'model_best.pth'))
            log.info(f"Save checkpoint at {os.path.join(model_dir, 'model_best.pth')}")
            state_dict = deepcopy(optimizer.state_dict())

            torch.save(
                {"epoch": epoch, 
                "optim_state_dict": state_dict,
                "best_metrics": best_cd_l1,
                "best_epoch": best_epoch_l1},
                os.path.join(optim_dir, 'optim_best.pth')
            )
            
        
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_last.pth'))
        log.info(f"Save checkpoint at {os.path.join(model_dir, 'model_last.pth')}")
        state_dict = deepcopy(optimizer.state_dict())
        torch.save(
            {"epoch": epoch, 
            "optim_state_dict": state_dict,
            "best_metrics": best_cd_l1,
            "best_epoch": best_epoch_l1},
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
