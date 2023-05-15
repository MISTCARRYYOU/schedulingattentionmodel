import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from agent_AM.nets.attention_model import set_decode_type
from agent_AM.utils.log_utils import log_values
from agent_AM.utils.functions import move_to
from agent_AM.problems.schedule.args import episode_record_dir

from agent_AM.problems.schedule.args import decode_type



import timeit


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost

# 下面这个函数挺耗费时间的
def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset())
    # 这个training dataset 是1280000 x dict(loc:10x2, demand:1x10, depot:1x2)，也就是1280000个instance

    # 下面这个是把上面的数据打包成batch size
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)  # num_workers>0时无法debug

    # Put model in train mode!
    model.train()  # 这里的model已经是继承nn.Module的AM模型或者PN模型了
    set_decode_type(model, decode_type)

    # 下面开始按照每一个batch进行训练, cvrp问题中共有2500个batches
    # for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
    for batch_id, batch in enumerate(training_dataloader):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("-------------------Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    start_va = time.time()
    avg_reward = validate(model, val_dataset, opts)
    end_va =time.time()
    print('-----------------------The validating time is {}'.format(end_va - start_va))

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)  # 从这可以看出来它的baseline其实就是上一个epoch的自己

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


# 这个是最后一层训练的rollout
def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):

    x, bl_val = baseline.unwrap_batch(batch)

    x = move_to(x, opts.device)  # cpu gpu 转换
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    if step % 10 == 0:
        is_record = True
        with open(episode_record_dir, 'a') as file:
            print('\n---{} step---'.format(step), file=file)
    else:
        is_record = False

    cost, log_likelihood = model(x, is_record=is_record)

    # # 这两个都在cpu上
    # cost = cost.cuda()
    # log_likelihood = log_likelihood.cuda()

    # Evaluate baseline, get baseline loss if any (only for critic)

    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    with open('./my_data_and_graph/SAM/loss.txt', 'a') as file:
        print('step: {} ; loss: {} ; average cost: {} '.format(step, loss, cost.mean()), file=file)

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    # 梯度对齐操作
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
