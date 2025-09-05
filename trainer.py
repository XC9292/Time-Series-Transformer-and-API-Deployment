
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
import torch.nn.functional as F
from model import TimeSeriesTransformer 
from dataset import SequentialDataset
import torch.backends.cudnn as cudnn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
sns.set_theme(rc={"figure.figsize":(18, 10)}) 

def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
    
def main_worker(cfg, args):
    
    fix_random_seeds(cfg['seed'])

    # Create TensorBoard writer
    log_dir = os.path.join(cfg['checkpoint']['ckpt_path'], 'tensorboard_logs').format(cfg['model']['type'], cfg['model']['window_size'])
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"=> TensorBoard logs will be saved to: {log_dir}")

    # create model
    print("=> creating model ...")
    model = TimeSeriesTransformer(
        input_dim=cfg['model']['in_dim'],
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['num_heads'],
        n_layers=cfg['model']['num_layers'],
        d_ff=cfg['model']['ff_dim'],
        max_seq_len=cfg['model']['window_size'],
        output_dim=cfg['model']['out_dim'],
        dropout=cfg['model']['dropout_rate']
    ).cuda()
    print("=> model created.")

    model_path = os.path.join(cfg['checkpoint']['ckpt_path']).format(cfg['model']['type'], cfg['model']['window_size'])
    
    
    # Data loading code
    root_dir = cfg['data']['root_path']

    train_dataset = SequentialDataset(
                                    json_file=os.path.join(root_dir, 'train_data_windows_{}.json'.format(cfg['model']['window_size'])),
                                    window_size=cfg['model']['window_size'],
                                    feature_columns=cfg['data']['feature_columns'],
                                       )

    test_dataset = SequentialDataset(
                                    json_file=os.path.join(root_dir, 'test_data_windows_{}.json'.format(cfg['model']['window_size'])),
                                    window_size=cfg['model']['window_size'],
                                    feature_columns=cfg['data']['feature_columns'],
                                       )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['data']['train_batch_size'], shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg['data']['test_batch_size'], shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()

    optim_params = model.parameters()

    optimizer = torch.optim.AdamW(optim_params, cfg['optimizer']['base_lr'],
                                  weight_decay=cfg['optimizer']['weight_decay'])
    
    lr_scheduler = cosine_scheduler(
        cfg['optimizer']['base_lr'],
        cfg['optimizer']['min_lr'],
        cfg['optimizer']['epochs'],
        len(train_loader),
        warmup_epochs=cfg['optimizer']['warmup_epochs'],
        start_warmup_value=cfg['optimizer']['warmup_lr']
    )

    cudnn.benchmark = True

    

    
    for epoch in range(cfg['optimizer']['epochs']):
        
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, cfg, args)
        
        # test for one epoch
        test_loss, CO2_pred, CO2_GT = test(test_loader, model, criterion, epoch, cfg, args)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log CO2 predictions vs ground truth plots for first 6 samples
        if epoch % 10 == 0:  # Log every 10 epochs to avoid cluttering
        
            # Convert tensors to numpy for plotting
            co2_pred_np = torch.cat(CO2_pred).cpu().numpy()
            co2_gt_np = torch.cat(CO2_GT).cpu().numpy()

            fig, ax = plt.subplots()

            # Plot Ground truth CO2 estimation
            ax.scatter(np.arange(co2_gt_np.shape[0]), co2_gt_np[:, 0], marker='*', color='b', s=30,label='Ground Truth@1')
            ax.scatter(np.arange(co2_gt_np.shape[0]), co2_gt_np[:, 1], marker='v', color='g', s=30,label='Ground Truth@2')


            # Plot prediction line (y=x)
            ax.plot(np.arange(co2_pred_np.shape[0]), co2_pred_np[:, 0], '--', color='b', linewidth=3,label='Prediction@1')
            ax.plot(np.arange(co2_pred_np.shape[0]), co2_pred_np[:, 1], '--', color='g', linewidth=3,label='Prediction@2')

            ax.set_xlabel('Time Steps')
            ax.set_ylabel('CO2 Estimations')
            ax.set_title(f'CO2 Predictions vs Ground Truth @1-2')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Log to TensorBoard
            writer.add_figure(f'CO2_Pred_vs_GT/Sample_1-2', fig, epoch)
            plt.close(fig)
            
            
            fig, ax = plt.subplots()

            # Plot Ground truth CO2 estimation

            ax.scatter(np.arange(co2_gt_np.shape[0]), co2_gt_np[:, 2], marker='>', color='r', s=30, label='Ground Truth@3')
            ax.scatter(np.arange(co2_gt_np.shape[0]), co2_gt_np[:, 3], marker='<', color='c', s=30, label='Ground Truth@4')

            # Plot prediction line (y=x)
            ax.plot(np.arange(co2_pred_np.shape[0]), co2_pred_np[:, 2], '--', color='r', linewidth=3, label='Prediction@3')
            ax.plot(np.arange(co2_pred_np.shape[0]), co2_pred_np[:, 3], '--', color='c', linewidth=3, label='Prediction@4')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('CO2 Estimations')
            ax.set_title(f'CO2 Predictions vs Ground Truth @3-4')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Log to TensorBoard
            writer.add_figure(f'CO2_Pred_vs_GT/Sample_3-4', fig, epoch)
            plt.close(fig)
            
            
            fig, ax = plt.subplots()

            # Plot Ground truth CO2 estimation
            ax.scatter(np.arange(co2_gt_np.shape[0]), co2_gt_np[:, 4], marker='*', color='m', s=30, label='Ground Truth@5')
            ax.scatter(np.arange(co2_gt_np.shape[0]), co2_gt_np[:, 5], marker='v', color='y', s=30, label='Ground Truth@6')

            # Plot prediction line 
            ax.plot(np.arange(co2_pred_np.shape[0]), co2_pred_np[:, 4], '--', color='m', linewidth=3, label='Prediction@5')
            ax.plot(np.arange(co2_pred_np.shape[0]), co2_pred_np[:, 5], '--', color='y', linewidth=3, label='Prediction@6')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('CO2 Estimations')
            ax.set_title(f'CO2 Predictions vs Ground Truth @5-6')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Log to TensorBoard
            writer.add_figure(f'CO2_Pred_vs_GT/Sample_5-6', fig, epoch)
            plt.close(fig)

        
        # Log model parameters histogram every 10 epochs
        if (epoch + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # Save model
            
        checkpoint_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }

        # Save checkpoint
        torch.save(checkpoint_dict, os.path.join(model_path, 'model.pth.tar'))

    # Close the writer
    writer.close()
    print(f"=> Training completed. TensorBoard logs saved to: {log_dir}")


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, cfg, args):
    print(f"\nTraining at Epoch: {epoch+1}\n")
    model.train()
    Train_Loader = tqdm(train_loader)
    
    total_loss = 0.0
    num_batches = 0
    
    for it, train_data in enumerate(Train_Loader):
        
        # update weight decay and learning rate according to their schedule
        it = len(train_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_scheduler[it]
            
        data_time_series, targets = train_data[0].cuda(), train_data[1].cuda()

        # forward pass
        predictions = model(data_time_series, use_causal_mask=True)
        Loss = criterion(predictions, targets[:, -1])
        
        # backward pass
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
        total_loss += Loss.item()
        num_batches += 1
        
        Train_Loader.set_postfix(loss=Loss.item(), epoch=epoch+1, avg_loss=total_loss/num_batches)
    
    avg_train_loss = total_loss / num_batches
    print(f"=> Training Epoch {epoch+1} completed. Average Loss: {avg_train_loss:.6f}")
    return avg_train_loss


def test(test_loader, model, criterion, epoch, cfg, args):
    print(f"\nTesting at Epoch: {epoch+1}\n")
    model.eval()
    Test_Loader = tqdm(test_loader)
    
    num_test_data = 0
    RMSE = 0
    CO2_pred = []
    CO2_GT = []
    with torch.no_grad():
        for i, test_data in enumerate(Test_Loader):
            scale = torch.tensor(cfg['data']['CO2_scale']).cuda()
            data_time_series, targets = test_data[0].cuda(), test_data[1].cuda()
            targets = targets * scale

            
            # forward pass
            if i == 0:
                for step in range(cfg['model']['window_size']):
                    predictions = model(data_time_series[:, :step+1, :], use_causal_mask=True) * scale
                    rmse = torch.sqrt(criterion(predictions, targets[:, step]))
                    CO2_pred.append(predictions)
                    CO2_GT.append(targets[:, step])
                    RMSE += rmse
                    num_test_data += 1
            else:
                predictions = model(data_time_series, use_causal_mask=True) * scale
                rmse = torch.sqrt(criterion(predictions, targets[:, step]))
                CO2_pred.append(predictions)
                CO2_GT.append(targets[:, step])
                RMSE += rmse
                num_test_data += 1

            Test_Loader.set_postfix(RMSE=RMSE/num_test_data, epoch=epoch+1)

    avg_test_rmse = RMSE / num_test_data
    print(f"=> Testing Epoch {epoch+1} completed. Average RMSE: {avg_test_rmse:.6f}")
    return avg_test_rmse, CO2_pred, CO2_GT