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
import Metrics
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

MAX = float("inf")

#######You can Modify her
B = 4
train_dir = '/user/viktor.tevosyan/u17320/project/LoDoPaB'
valid_dir = '/user/viktor.tevosyan/u17320/project/LoDoPaB'
best_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/best.pth'
latest_model_checkpoint = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/last.pth'
training_results_dir = '/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/'


#######


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)
    train_set = CT_library.LoDoPaB_Dataset(sino_dir=train_dir, gt_images_dir=train_dir, suffix='train',
                                           amount_images=1000)
    valid_set = CT_library.LoDoPaB_Dataset(sino_dir=valid_dir, gt_images_dir=valid_dir, suffix='valid',
                                           amount_images=1000)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_set, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(dataset=train_set, batch_size=B, shuffle=False, pin_memory=True, sampler=train_sampler,
                              num_workers=2)
    valid_loader = DataLoader(dataset=valid_set, batch_size=B, shuffle=False, pin_memory=True, sampler=valid_sampler,
                              num_workers=2)

    ## Here will the training probably happen

    windowII = Modells.LearnableWindowII()
    ramp_filter = Reconstructor.Ramp_Filter()
    filtering_module = Reconstructor.Filtering_Module(filter_model=ramp_filter, window_model=windowII)

    vanilla_backproj = Reconstructor.Vanilla_Backproj()

    no_post = torch.nn.Identity()


    reconstructor = Reconstructor.LearnableFBP(filtering_module=filtering_module,
                                               backprojection_module=vanilla_backproj, post_processing_module=no_post)
    reconstructor.load_state_dict(torch.load('/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/best15.pth', map_location='cpu')['model_state_dict'])
    reconstructor = reconstructor.to(rank)
    reconstructor = DDP(reconstructor, device_ids=[rank])

    optimizer = torch.optim.Adam(reconstructor.module.filtering_module.window_model.parameters(), lr=1e-4)
    optimizer.load_state_dict(torch.load('/user/viktor.tevosyan/u17320/project/Codebase/windowII+edge+perceptual+subset/best15.pth', map_location='cpu')['optimizer_state_dict'])
    l1_loss = torch.nn.L1Loss()
    ms_ssim = piq.MultiScaleSSIMLoss().to(rank)
    ge = Criterion.GradientEdgeLoss().to(rank)


    loss_fn = lambda x,y : 0.25*l1_loss(x,y) + 1*ms_ssim(torch.clamp(x, 0, 1),torch.clamp(y, 0, 1)) + 1*ge(x,y)

    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1)

    trainer = Trainer.Trainer(model=reconstructor, train_loader=train_loader, valid_loader=valid_loader,
                              optim=optimizer,
                              criterion=loss_fn, criterion_sinogram=None, best_model_checkpoint=best_model_checkpoint,
                              latest_model_checkpoint=latest_model_checkpoint, lr_scheduler=scheduler, rank=rank,
                              one_cycle_lr=False, training_results_dir=training_results_dir)

    losses = trainer.train(10)

    

    
    if rank == 0:
        torch.save(losses, training_results_dir + 'losses20.pth')




    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.set_start_method("fork", force=True)
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


