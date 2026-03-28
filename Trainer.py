import torch
import os

import CT_library
import Metrics
import torch.distributed as dist

MAX = float("inf")


class Trainer:
    """
    Manages distributed training and evaluation of a CT reconstruction model.

    Training runs via PyTorch DistributedDataParallel (DDP). Each process owns
    one `rank`; only rank 0 writes checkpoints and accumulates loss histories.

    Supports an optional sinogram-domain auxiliary loss computed in Radon space,
    two LR-scheduler modes (OneCycleLR per batch, ReduceLROnPlateau per epoch),
    and periodic epoch-level checkpoint saves every 5 epochs.

    Args:
        model: DDP-wrapped model to train.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        optim (Optimizer): PyTorch optimizer (e.g. Adam).
        criterion: Primary loss function applied in image space (e.g. MSELoss).
        criterion_sinogram (optional): Additional loss function applied in Radon
            (sinogram) space. Set to None to disable.
        best_model_checkpoint (str, optional): File path to save the best model
            checkpoint whenever validation loss improves.
        latest_model_checkpoint (str, optional): Reserved for future use.
        lr_scheduler (optional): Learning-rate scheduler. Pass a OneCycleLR
            instance together with ``one_cycle_lr=True``, or a
            ReduceLROnPlateau instance otherwise.
        rank: Device rank (GPU index or ``torch.device('cpu')``). Default: cpu.
        one_cycle_lr (bool): If True, steps the scheduler after every batch
            instead of after every epoch. Default: False.
        training_results_dir (str, optional): Directory for periodic epoch
            checkpoints saved every 5 epochs.
    """
    def __init__(self, model, train_loader, valid_loader, optim, criterion, criterion_sinogram=None, best_model_checkpoint=None,
                 latest_model_checkpoint=None, lr_scheduler=None, rank=torch.device('cpu'), one_cycle_lr=False, training_results_dir=None):
        print("Initializing Variables")
        # Init vars
        self.rank = rank
        self.optimizer = optim
        self.best_model_checkpoint = best_model_checkpoint
        self.latest_model_checkpoint = latest_model_checkpoint
        self.training_results_dir = training_results_dir
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.one_cycle_lr = one_cycle_lr
        self.criterion_sinogram = criterion_sinogram

        self.model = model
        self.criterion = criterion

        self.radon = CT_library.RadonTransform(rank).radon
        print("Initing result arrays")
        self.train_losses = []
        self.val_losses = []
        self.metrics = []

        self.val_best = MAX
        self.stop_training = False
        self.early_stopper_counter = 0
        print("Trainer init completed")


    def _run_train_batch(self, source, targets):
        """
        Performs a single training step on one batch.

        Computes the combined image-space and optional sinogram-space loss,
        runs backpropagation, and updates model weights. When ``one_cycle_lr``
        is enabled the LR scheduler is stepped here after each batch.

        Args:
            source (Tensor): Input batch (e.g. filtered back-projection), shape (B, C, H, W).
            targets (Tensor): Ground-truth batch, shape (B, C, H, W).

        Returns:
            float: Scalar total loss for this batch.
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss1 = self.criterion(output, targets)
        loss2 = torch.tensor(0.0, device=self.rank)
        if self.criterion_sinogram is not None:
            targets_sino = self.radon(targets).to(self.rank)
            output_sino = self.radon(output).to(self.rank)
            loss2 = self.criterion_sinogram(targets_sino, output_sino)


        loss = loss1 + loss2

        loss.backward()
        self.optimizer.step()

        if self.one_cycle_lr:
            self.lr_scheduler.step()

        return loss.item()

    def _run_valid_batch(self, source, targets):
        """
        Computes the loss for a single validation batch without gradient tracking.

        Args:
            source (Tensor): Input batch, shape (B, C, H, W).
            targets (Tensor): Ground-truth batch, shape (B, C, H, W).

        Returns:
            float: Scalar total loss for this batch.
        """
        output = self.model(source)
        loss1 = self.criterion(output, targets)
        loss2 = torch.tensor(0.0, device=self.rank)
        if self.criterion_sinogram is not None:
            targets_sino = self.radon(targets).to(self.rank)
            output_sino = self.radon(output).to(self.rank)
            loss2 = self.criterion_sinogram(targets_sino, output_sino)

        loss = loss1 + loss2
        return loss.item()

    def _run_train_epoch(self, epoch):
        """
        Runs one full training epoch over all batches.

        Sets the model to train mode, iterates over the train loader, and
        aggregates the per-batch losses across all DDP ranks via all_reduce.
        Only rank 0 appends the averaged epoch loss to ``self.train_losses``.

        Args:
            epoch (int): Zero-based epoch index (used for logging and sampler).
        """
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.rank}] Training | Epoch: {epoch + 1} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        running_loss = 0
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        for num, (source, targets) in enumerate(self.train_loader):
            source = source.to(self.rank)
            targets = targets.to(self.rank)
            running_loss += self._run_train_batch(source, targets)

        total_loss = torch.tensor(running_loss).to(self.rank)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)  # Sum losses from all GPUs
        epoch_loss = (total_loss / (len(self.train_loader) * dist.get_world_size())).item()

        if self.rank == 0:  # Only store on master process
            self.train_losses.append(epoch_loss)

        print(f"[GPU{self.rank}] Training | Epoch: {epoch + 1} finished with loss: {epoch_loss}")

    def _run_valid_epoch(self, epoch):
        """
        Runs one full validation epoch over all batches.

        Sets the model to eval mode and disables gradient computation. Losses
        are aggregated across DDP ranks via all_reduce. On rank 0, the epoch
        loss is appended to ``self.val_losses`` and, if it is the best seen so
        far, the model is saved to ``best_model_checkpoint``. For
        ReduceLROnPlateau schedulers the step is called here with the epoch
        loss; OneCycleLR is handled per-batch in ``_run_train_batch``.

        Args:
            epoch (int): Zero-based epoch index (used for logging).
        """
        b_sz = len(next(iter(self.valid_loader))[0])
        print(f"[GPU{self.rank}] Validation | Epoch: {epoch + 1} | Batchsize: {b_sz} | Steps: {len(self.valid_loader)}")
        running_loss = 0

        self.model.eval()
        with torch.no_grad():
            for num, (source, targets) in enumerate(self.valid_loader):
                source = source.to(self.rank)
                targets = targets.to(self.rank)
                running_loss += self._run_valid_batch(source, targets)

        total_loss = torch.tensor(running_loss).to(self.rank)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)  # Sum losses from all GPUs
        epoch_loss = (total_loss / (len(self.valid_loader) * dist.get_world_size())).item()

        if not self.one_cycle_lr and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_loss)

        if self.rank == 0:
            self.val_losses.append(epoch_loss)
            if epoch_loss < self.val_best:
                self._save_checkpoint(self.best_model_checkpoint)
                self.val_best = epoch_loss

        print(f"[GPU{self.rank}] Validation | Epoch: {epoch + 1} finished with loss: {epoch_loss}")

    def _save_checkpoint(self, checkpoint_dir):
        """
        Saves the model and optimizer state to disk.

        If a file already exists at ``checkpoint_dir`` it is removed first so
        that the file is always replaced atomically. Only call this from rank 0.

        Args:
            checkpoint_dir (str): Target file path for the checkpoint (.pth).
        """
        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_best': self.val_best
        }
        if os.path.exists(checkpoint_dir):
            os.remove(checkpoint_dir)
        torch.save(checkpoint, checkpoint_dir)

    def train(self, max_epochs: int):
        """
        Runs the full training loop for up to ``max_epochs`` epochs.

        On rank 0, a periodic checkpoint is saved every 5 epochs (starting
        from epoch 5). Training can be stopped early by setting
        ``self.stop_training = True`` from outside the loop.

        Args:
            max_epochs (int): Maximum number of epochs to train for.

        Returns:
            dict: ``{'train_losses': list, 'valid_losses': list}`` containing
                the per-epoch losses collected on rank 0.
        """
        for epoch in range(max_epochs):

            ###
            if self.rank == 0:
                if epoch > 0 and epoch % 5 == 0:
                    dir_to_save = self.training_results_dir + f'checkpoint_epoch_{epoch}.pth'
                    self._save_checkpoint(dir_to_save)
            ###



            if self.stop_training:
                break

            print(f"Entering Epoch {epoch + 1}")
            self._run_train_epoch(epoch)
            self._run_valid_epoch(epoch)
            print(f"End of Epoch {epoch + 1}")
        return {'train_losses': self.train_losses, 'valid_losses': self.val_losses}

    def evaluate(self):
        """
        Evaluates the model on the full validation set and stores per-sample metrics.

        Runs inference without gradients, collects PSNR, L1, MSE, and SSIM for
        every sample, then aggregates them across all DDP ranks via all_reduce.
        Results are stored in ``self.metrics`` as a dict containing mean, best
        (with index), and worst (with index) for each metric.

        Note: SSIM is computed on clamped outputs in [0, 1].
        """
        print(f"Entering Evaluation")
        self.model.eval()

        ls_psnr, ls_l1, ls_mse, ls_ssim = [], [], [], []

        with torch.no_grad():
            for num, (source, targets) in enumerate(self.valid_loader):
                source = source.to(self.rank)
                outputs = self.model(source)
                targets = targets.to(self.rank)
                psnr, l1, mse, ssim = self._evaluator(outputs, targets)
                ls_psnr.append(psnr)
                ls_l1.append(l1)
                ls_mse.append(mse)
                ls_ssim.append(ssim)

        ls_psnr = torch.stack(ls_psnr)
        ls_l1 = torch.stack(ls_l1)
        ls_mse = torch.stack(ls_mse)
        ls_ssim = torch.stack(ls_ssim)

        dist.all_reduce(ls_psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(ls_l1, op=dist.ReduceOp.SUM)
        dist.all_reduce(ls_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(ls_ssim, op=dist.ReduceOp.SUM)

        world_size = dist.get_world_size()
        ls_psnr /= world_size
        ls_l1 /= world_size
        ls_mse /= world_size
        ls_ssim /= world_size

        self.metrics = {'mean_psnr': ls_psnr.mean(), 'mean_l1': ls_l1.mean(), 'mean_mse': ls_mse.mean(),
                        'mean_ssim': ls_ssim.mean(),
                        'best_psnr': (ls_psnr.max(), ls_psnr.argmax()), 'best_l1': (ls_l1.min(), ls_l1.argmin()),
                        'best_mse': (ls_mse.min(), ls_mse.argmin()), 'best_ssim': (ls_ssim.max(), ls_ssim.argmax()),
                        'worst_psnr': (ls_psnr.min(), ls_psnr.argmin()), 'worst_l1': (ls_l1.max(), ls_l1.argmax()),
                        'worst_mse': (ls_mse.max(), ls_mse.argmax()), 'worst_ssim': (ls_ssim.min(), ls_ssim.argmin())}

    def _evaluator(self, pred, target):
        """
        Computes all four image quality metrics for a single batch.

        SSIM is evaluated on outputs clamped to [0, 1] to match the expected
        value range of the metric implementation.

        Args:
            pred (Tensor): Model output, shape (B, C, H, W).
            target (Tensor): Ground-truth image, shape (B, C, H, W).

        Returns:
            tuple: (psnr, l1, mse, ssim) — one scalar tensor per metric.
        """
        psnr = Metrics.psnr(pred, target)
        l1 = Metrics.l1_loss(pred, target)
        mse = Metrics.mse_loss(pred, target)
        ssim = Metrics.ssim_metric(torch.clamp(pred, 0, 1), torch.clamp(target, 0, 1))

        return psnr, l1, mse, ssim
