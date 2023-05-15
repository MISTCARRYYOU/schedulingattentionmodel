'''
    这个文件是自己用于设置SAM参数的地方
'''


import torch

# cost 参数
kf1 = 5.0
kf2 = 5.0   # 减小波动影响，与之相关的都乘了0.1
kf3 = 2.0


max_time_scaled = 25
waiting_time_piece = 4

# problem的参数
orders_number = 40  # 订单数量
unified_order_size = 10  # 订单的尺寸
enterprise_size = 20  # 公司的数量
service_lib_size = 16  # 服务的类型数
num_samples = 1024
logistic_time_unit = 5.0  # 单位距离的物流时间花费
logistic_expenditure_unit = 10.0  # 单位距离的物流价格花费
decode_type = 'greedy'
# 决定cost的三组参数，总比例在2:2:1
W_t = 2
W_e = 2
W_r = 1
# 随机种子设置为1  同一个随机种子下结果是可以复现的（这个种子主要是针对的numpy的随机性）
# seed = 1
seed = 1
is_use_GPU = False
is_save_input = False  # 是否保存input，如果保存的话可能会占运行时间
episode_record_dir = './my_data_and_graph/SAM/record.txt'
# sample_root_path = './my_data_and_graph/scene_samples/dataset-T400-seed'
# 下面这个参数不仅是保存的，也是用于读取的
input_save_path = './my_data_and_graph/scene_samples/input-T100-seed'  # exp-1
# input_save_path = './my_data_and_graph/scene_samples/generalizationtest/sucai/input-sucai-T200-seed'  # exp-2
# input_save_path = './my_data_and_graph/scene_samples/scalablity/sucai/input-sucai-T'  # exp-3
if is_use_GPU:
    init_pos = torch.tensor([0, 0], dtype=torch.float32, device='cuda:0')
else:
    init_pos = torch.tensor([0, 0], dtype=torch.float32)
# load_model = './condact_experiments/pretrainedmodels/T400/SAM/model0.pt'  # 选择加载哪个模型
load_model = 0  # 0 代表不加载模型，那么此时无法评估，只能重新训练
