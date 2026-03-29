import CT_library
import Reconstructor
import Criterion
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import Trainer
import piq
import Modells

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

MAX = float("inf")

#######You can Modify her
B = 4
train_dir = '/user/viktor.tevosyan/u17320/project/LoDoPaB'
valid_dir = '/user/viktor.tevosyan/u17320/project/LoDoPaB'
best_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/UNet_with_proposed_loss/best.pth'
latest_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/UNet_with_proposed_loss/last.pth'
training_results_dir = '/user/viktor.tevosyan/u17320/project/Codebase/UNet_with_proposed_loss/'


#######


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)
    train_set = CT_library.LoDoPaB_Dataset(sino_dir=train_dir, gt_images_dir=train_dir, suffix='train')
    valid_set = CT_library.LoDoPaB_Dataset(sino_dir=valid_dir, gt_images_dir=valid_dir, suffix='valid')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_set, num_replicas=world_size, rank=rank
    )
    # shuffle done by DS, pin_memory=True speeds up the transfer between CPU and GPU
    train_loader = DataLoader(dataset=train_set, batch_size=B, shuffle=False, pin_memory=True, sampler=train_sampler,
                              num_workers=2)
    valid_loader = DataLoader(dataset=valid_set, batch_size=B, shuffle=False, pin_memory=True, sampler=valid_sampler,
                              num_workers=2)

    ## Here will the training probably happen

    windowII = Modells.LearnableWindowII()
    ramp_filter = Reconstructor.Ramp_Filter()
    filtering_module = Reconstructor.Filtering_Module(filter_model=ramp_filter, window_model=windowII)

    vanilla_backproj = Reconstructor.Vanilla_Backproj()

    unet = Modells.UNet(1)

    reconstructor = Reconstructor.LearnableFBP(filtering_module=filtering_module,
                                               backprojection_module=vanilla_backproj, post_processing_module=unet)

    # WE dont need the rest, only the post-processing UNet
    for param in reconstructor.filtering_module.parameters():
        param.requires_grad = False

    for param in reconstructor.backprojection_module.parameters():
        param.requires_grad = False

    reconstructor = reconstructor.to(rank)
    reconstructor = DDP(reconstructor, device_ids=[rank])

    optimizer = torch.optim.Adam(reconstructor.module.post_processing_module.parameters(), lr=1e-4)
    l1_loss = torch.nn.L1Loss().to(rank)
    mssim = piq.MultiScaleSSIMLoss().to(rank)
    ge = Criterion.GradientEdgeLoss().to(rank)
    vgg = Criterion.PerceptualLoss().to(rank)

    alpha = 1.0     # L1 or L2 anchor
    beta  = 0.5     # MS-SSIM (structural/perceptual)
    gamma = 0.1     # gradient/edge loss (preserve edges)
    delta = 0.05    # VGG perceptual (feature) loss — very small for medical data

    loss = lambda x,y : alpha*l1_loss(x,y) + beta*mssim(x,y) + gamma*ge(x,y) + delta*vgg(x,y)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1)

    trainer = Trainer.Trainer(model=reconstructor, train_loader=train_loader, valid_loader=valid_loader,
                              optim=optimizer,
                              criterion=loss, criterion_sinogram=None, best_model_checkpoint=best_model_checkpoint,
                              latest_model_checkpoint=latest_model_checkpoint, lr_scheduler=scheduler, rank=rank,
                              one_cycle_lr=False, training_results_dir=training_results_dir)


    losses = trainer.train(5)
    trainer.evaluate()
    if rank == 0:
        torch.save(losses, training_results_dir + 'losses5.pth')
        torch.save(trainer.metrics, training_results_dir + 'metrics5.pth')

    losses = trainer.train(5)
    trainer.evaluate()
    if rank == 0:
        torch.save(losses, training_results_dir + 'losses10.pth')
        torch.save(trainer.metrics, training_results_dir + 'metrics10.pth')

    losses = trainer.train(5)
    trainer.evaluate()
    if rank == 0:
        torch.save(losses, training_results_dir + 'losses15.pth')
        torch.save(trainer.metrics, training_results_dir + 'metrics15.pth')

    losses = trainer.train(5)
    trainer.evaluate()
    if rank == 0:
        torch.save(losses, training_results_dir + 'losses20.pth')
        torch.save(trainer.metrics, training_results_dir + 'metrics20.pth')

    losses = trainer.train(5)
    trainer.evaluate()
    if rank == 0:
        torch.save(losses, training_results_dir + 'losses25.pth')
        torch.save(trainer.metrics, training_results_dir + 'metrics25.pth')

    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.set_start_method("fork", force=True)
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


