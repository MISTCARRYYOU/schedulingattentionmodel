from agent_AM.problems.schedule.state_schedule import StateSchedule
from torch.utils.data import Dataset
import torch
import numpy as np
from agent_AM.problems.schedule.args import *
import pickle


# 整个调度问题的环境类
class Schedule(object):

    NAME = 'schedule'

    @staticmethod
    def make_dataset(*args, **kwargs):
        return ScheduleDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSchedule.initialize(*args, **kwargs)


# 生成数据集，因为需要以DL的方式来进行训练
class ScheduleDataset(Dataset):

    def __init__(self, orders_number=orders_number, unified_order_size=unified_order_size, enterprise_size=enterprise_size, service_lib_size=service_lib_size, num_samples=num_samples,seed=seed):
        super(ScheduleDataset, self).__init__()

        self.data_set = []

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 生成数据集样本，主要是企业的分布位置、能力、以及企业包含的服务能力。
        self.data = []
        for _ in range(num_samples):
            temp_service = torch.zeros(enterprise_size, service_lib_size)
            temp_service[torch.rand(enterprise_size, service_lib_size) > 0.5] = 1
            # 为保证整个环境中不会出现订单所需资源不存在的情况：
            assert enterprise_size >= service_lib_size, '多设置点enterprise吧，不要少于service lib size'
            for index in range(service_lib_size):
                temp_service[index][index] = 1
            # ---------------------------------------

            all_order_lists = [np.random.choice(range(service_lib_size),
                               unified_order_size, replace=False).tolist()
                               for _i in range(orders_number)]

            # 将以上多个list合成一个大list，注意此处没有用0作为间隔，判断当前订单已经完成的标志只有靠state的cur_task
            temp_order_list = []
            for eve_list in all_order_lists:
                temp_order_list += eve_list

            # 服务的容量
            service_capacity = torch.randint(5, 10, [enterprise_size, service_lib_size])

            temp_dict = {
                    # 公司的相关信息
                    'loc': torch.FloatTensor(enterprise_size, 2).uniform_(0.1, 1),
                    'services': temp_service,
                    'service_capacity': temp_service*service_capacity,
                    'abl': torch.FloatTensor(enterprise_size).uniform_(1, 3),
                    'depot': torch.tensor([0, 0], dtype=torch.float),
                    # 服务的相关信息
                    'service_time': torch.randint(5, 20, [service_lib_size]),
                    'service_expenditure': torch.randint(5, 21, [service_lib_size]),
                    # 订单的相关信息(主要是依赖的服务顺序) 代表开始下一个订单
                    'order': torch.tensor(temp_order_list),  # [ 1 3 4 7 9 8 1 ...]，
                    'order_splited': torch.tensor(all_order_lists)  # 这个二维的用来观察一个order的结束
                }

            # 计算这个instance下的最大值
            max_distance = 1.2  # (1,1) -> (0.1,0.1)
            temp_time = 0
            temp_expenditure = 0
            temp_reliability = 0
            order_length = temp_dict['order'].shape[0]  # 有几个order
            # 先线性完成所有的服务，不考虑物流
            for eve_service in temp_dict['order']:
                temp_time += temp_dict['service_time'][eve_service] / (kf1 + 1)  # 1 是最低的abl
                temp_expenditure += temp_dict['service_expenditure'][eve_service] * (kf2 + 3*0.1)
                temp_reliability += kf3 * 3  # 3 是abl的最大值

            # 开始计算物流
            temp_time += max_distance * (order_length + orders_number + 1) * logistic_time_unit
            # print(temp_expenditure.dtype, max_distance.dtype, logistic_expenditure_unit.dtype)
            temp_expenditure += max_distance * logistic_expenditure_unit

            temp_dict['max_time'] = temp_time/max_time_scaled   # 时间的最大值差太多了
            temp_dict['max_expenditure'] = temp_expenditure
            temp_dict['max_reliability'] = temp_reliability

            self.data.append(
                temp_dict
            )
        self.size = len(self.data)

        # 将样本保存
        print('A dataset has been constructed and saved !!')
        # with open(sample_root_path + str(seed) + '.pkl', 'wb') as pkf:
        #     pickle.dump(self.data, pkf)
        # print(self.data[0]['order_splited'])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
