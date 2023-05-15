'''
将AM模型接入调度场景

'''

import os
import json
import pprint as pp

import torch
import torch.optim as optim
# from tensorboard_logger import Logger as TbLogger
import time
from agent_AM.nets.critic_network import CriticNetwork
from agent_AM.options import get_options
from agent_AM.train import train_epoch, validate, get_inner_model
from agent_AM.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from agent_AM.nets.attention_model import AttentionModel
from agent_AM.nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from agent_AM.utils.functions import torch_load_cpu, load_problem
from agent_AM.problems.schedule.args import decode_type
from agent_AM.nets.attention_model import set_decode_type

from agent_AM.problems.schedule.args import episode_record_dir, input_save_path, seed


def extract_from_input(inputs, chosen_instance):
    return {
        name: torch.cat((
            inputs[name][chosen_instance][None, :] if len(inputs[name].shape) > 1 else inputs[name][
            chosen_instance].unsqueeze(0), inputs[name][chosen_instance+1][None, :] if len(inputs[name].shape) > 1 else inputs[name][
            chosen_instance+1].unsqueeze(0)
        ))
        for name in inputs.keys()
    }


# 为多进程封装的训练函数
def _train(opts, model, optimizer,baseline,lr_scheduler, val_dataset, problem, tb_logger):

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_dataset,
            problem,
            tb_logger,
            opts
        )


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    # if not opts.no_tensorboard:
    #     tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)

    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        derict_test = opts.derict_test
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    if opts.derict_test:  # 直接对数据集进行测试
        assert opts.load_input is not None
        import pickle
        import pandas as pd
        import numpy as np
        from agent_AM.problems.schedule.args import W_e, W_r, W_t

        with open(opts.load_input, 'rb') as pkf:
            inputs = pickle.load(pkf)

        set_decode_type(model, decode_type)

        for_table = [['algorithms','cost', 'time-scaled', 'time', 'expenditure-scaled','expenditure', 'reliability-scaled', 'reliability', 'run time']]
        for_figure = [['No.', 'Algorithms', 'Cost']]
        for chosen in range(0,6,1):
            t1 = time.time()
            # input0 = extract_from_input(inputs, chosen)  # 选择和DRL一样的样本
            input0 = extract_from_input(inputs, chosen)  # 选择和DRL一样的样本
            costs, _ = model(input0)
            t = time.time()-t1
            print('计算{}个场景下的调度解共用时{}秒'.format(input0['loc'].shape[0], t))
            print('costs:', costs)
            # 下面将costs的内容保存到excel中  save_excel_path_for_table/figure
            cost_cal = W_t*costs[0][0] + W_e*costs[0][2] - W_r*costs[0][4]
            for_table.append(['SAM', cost_cal] + costs[0] + [t/2])  # [总cost,各个部分的占比, 一个样本的时间]
            for_figure.append([chosen, 'SAM', cost_cal])

        pd.DataFrame(for_figure).to_excel(opts.save_excel_path_for_figure, header=False, index=0)
        pd.DataFrame(for_table).to_excel(opts.save_excel_path_for_table, header=False, index=0)

        return

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        # pass 在这句话中会进行评估验证工作
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    print('\n\nNow bigin training or testing\n\n')

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(opts.lr_decay ** epoch, 1e-5))

    # Start the actual training loop
    val_dataset = problem.make_dataset()

    if opts.resume:  # 这个是用于训练一半之后再接着训练用的
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1


    if opts.eval_only:
        validate(model, val_dataset, opts)
        # return None
    else:
        # return opts, model, optimizer, baseline, lr_scheduler, val_dataset, problem, tb_logger

        # # 这一部分是训练的部分
        # for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        #     train_epoch(
        #         model,
        #         optimizer,
        #         baseline,
        #         lr_scheduler,
        #         epoch,
        #         val_dataset,
        #         problem,
        #         tb_logger,
        #         opts
        #     )

        # 下面是开始尝试的多进程训练部分
        _train(opts, model, optimizer, baseline, lr_scheduler, val_dataset, problem, tb_logger)


if __name__ == "__main__":

    with open('./my_data_and_graph/SAM/loss.txt', 'w') as file:
        pass

    with open(episode_record_dir, 'w') as file2:
        pass
    opts = get_options()

    opts.derict_test = False  # 直接对数据集进行测试

    opts.load_input = input_save_path + str(seed) + '.pkl'  # 选择哪一个场景（数据集进行测试）# exp-1

    # opts.save_excel_path_for_table = './condact_experiments/exp5-1/data/T400-allin.xlsx'
    # opts.save_excel_path_for_figure = './condact_experiments/exp5-1/data/T400.xlsx'
    # opts.save_excel_path_for_table = './condact_experiments/exp5-1/data/T400-allin.xlsx'
    # pos, ser-time, ser-expen, ser-rel, \\ order-type,

    run(opts)

    # import torch.multiprocessing as mp
    #
    # if res is not None:
    #     opts, model, optimizer, baseline, lr_scheduler, val_dataset, problem, tb_logger = res
    #     num_processes = 1  # 设置4个进程
    #     model.share_memory()
    #     processes = []
    #     for rank in range(num_processes):
    #         # 4 个进程，每个进程epoch为150，也就是说其实迭代了 4*150 = 600 次 !!!
    #         p = mp.Process(target=_train,
    #                        args=(opts, model, optimizer, baseline, lr_scheduler, val_dataset, problem, tb_logger))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
