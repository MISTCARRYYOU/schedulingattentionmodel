'''
这个文件是环境修改最重要的文件，注意所有的操作都是基于batch的
'''

import torch
from typing import NamedTuple
from agent_AM.problems.schedule.args import waiting_time_piece, orders_number, kf1, kf2, kf3, \
    init_pos, is_use_GPU, logistic_expenditure_unit, logistic_time_unit, W_e, W_r, W_t


class StateSchedule(NamedTuple):

    # 公司信息
    coords: torch.Tensor  # Depot + Enterprise
    enterprise_services: torch.Tensor  # 每个公司拥有的service数量
    enterprise_abl: torch.Tensor  # 每个公司的能力值

    # 服务信息
    service_time: torch.Tensor
    service_expenditure: torch.Tensor
    service_capacity: torch.Tensor

    order: torch.Tensor
    order_split_index_flag: torch.Tensor

    # 索引信息
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    i: torch.Tensor  # Keeps track of step

    # 当前信息
    cur_coord: torch.Tensor  # 当前位置
    cur_time_cost: torch.Tensor  # 当前的时间消耗
    cur_expenditure_cost: torch.Tensor  # 当前的成本消耗
    cur_finished_task: torch.Tensor  # 当前完成的所有order的子任务
    prev_selected: torch.Tensor  # 记录之前选择的历史

    next_is_processing: torch.Tensor  # 用于判断下一步是否需要进行加工，不加工可能是加工完了、返回depot两种情况

    # for time mask
    cur_time_record: list  # [[time begin, time end], [time begin, time end], ..., ...]
    cur_time: torch.Tensor  # [current_time, ...]

    batch_time_cost: list
    batch_expenditure_cost: list
    batch_reliability_cost: list

    temp_order_time_cost: list
    temp_order_expenditure: list
    temp_order_reliability: list

    # 以下三个部分是为归一化做准备的变量
    max_time: torch.Tensor
    max_expenditure: torch.Tensor
    max_reliability: torch.Tensor

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    @staticmethod
    def initialize(input):
        """
        从input中提取信息组成state，所以input需要包含问题全部的信息
        """

        depot = input['depot']  # 出发点的坐标
        loc = input['loc']  # 所有公司的坐标

        order_split = input['order_splited']  # [[[],[]],...] batch size * m *n 三维
        order_flags = []
        for eve_batch in order_split:
            temp_flags = []
            for eve_order in eve_batch:
                if len(temp_flags) == 0:
                    temp_flags.append(eve_order.size(0)-1)
                else:
                    temp_flags.append(eve_order.size(0) + temp_flags[-1])
            order_flags.append(temp_flags)

        batch_size, n_loc, _ = loc.size()
        orders_size = input['order'].size(-1)

        # 下面返回的这些都是一个batch的
        return StateSchedule(
            coords=torch.cat((depot[:, None, :], loc), -2),
            enterprise_services=input['services'],  # 每个公司拥有的service数量
            enterprise_abl=input['abl'],  # 每个公司的能力值

            # batch_size * service_lib_size
            service_time=input['service_time'],
            service_expenditure=input['service_expenditure'],
            service_capacity=input['service_capacity'],

            order=input['order'],
            order_split_index_flag=torch.tensor(order_flags),  # batch_size * order size

            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps

            cur_coord=input['depot'][:, None, :],  # Add step dimension
            cur_time_cost=torch.zeros(batch_size, 1, device=loc.device),
            cur_expenditure_cost=torch.zeros(batch_size, 1, device=loc.device),
            # 用于指向当前正在加工的任务，如果遇到零直接跳过
            cur_finished_task=torch.zeros(batch_size, dtype=torch.int64) - 1,  # 全部初始化成-1
            prev_selected=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),

            cur_time_record=[[[] for j in range(n_loc)] for i in range(batch_size)],
            cur_time=torch.zeros(batch_size, dtype=torch.float32,device=loc.device),
            next_is_processing=torch.ones(batch_size, dtype=torch.bool),

            # 每个batch中的list用于存储以order为单位的cost
            # Time Expenditure Reliability
            batch_time_cost=[[] for _ in range(batch_size)],
            batch_expenditure_cost=[[] for _ in range(batch_size)],
            batch_reliability_cost=[[] for _ in range(batch_size)],

            temp_order_time_cost=[0 for _ in range(batch_size)],
            temp_order_expenditure=[0 for _ in range(batch_size)],
            temp_order_reliability=[0 for _ in range(batch_size)],

            max_time=input['max_time'],
            max_expenditure=input['max_expenditure'],
            max_reliability=input['max_reliability']
        )

    # selected是512，即一个batch的
    # self.的是上一次的，不带self.的是这次的
    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if a state represents a single step"

        batch_size = selected.size(0)

        selected = selected[:, None]  # Add dimension for step

        cur_coord = self.coords[self.ids, selected]  # ids 代表的是每个instance在batch中的位置             b x 1 x 2

        # 当前完成的任务指针加一
        # cur_finished_task = self.cur_finished_task + torch.ones(batch_size, dtype=torch.int64)                 # b

        # 计算cost，这块张量的分解选择是有问题的，先放着,意思反正在这，程序目前肯定是错了
        # 不用非要追求高阶tensor编程

        for batch_i in range(batch_size):
            # 已经结束的先过掉
            if self.cur_finished_task[batch_i] == len(self.order[batch_i]) - 1:  # 证明这个batch已经结束了调度
                # 只收集上述信息
                # 最后一步将收集的信息存入
                if self.temp_order_time_cost[batch_i] != 0 and self.temp_order_expenditure[batch_i] != 0:  # 他们两个一定是同时为0或者同时不为0的
                    # Time
                    self.batch_time_cost[batch_i].append(self.temp_order_time_cost[batch_i])
                    self.temp_order_time_cost[batch_i] = 0
                    # Exp
                    self.batch_expenditure_cost[batch_i].append(self.temp_order_expenditure[batch_i])
                    self.temp_order_expenditure[batch_i] = 0
                    # Rel
                    self.batch_reliability_cost[batch_i].append(self.temp_order_reliability[batch_i])
                    self.temp_order_reliability[batch_i] = 0
                continue

            # 选中要去的服务和公司
            selected_enterprise_in_batch_i = selected[batch_i][0]  # 公司
            if selected_enterprise_in_batch_i != 0:
                selected_enterprise_abl_in_batch_i = self.enterprise_abl[batch_i][selected_enterprise_in_batch_i-1]  # 公司能力
            else:
                selected_enterprise_abl_in_batch_i = 0  # 因为除以0会报错，因此用其纠错
            selected_service_in_batch_i = self.order[batch_i][self.cur_finished_task[batch_i]]  # 马上要处理的服务

            # 如果选的是depot并且下一个要完成的task并不在self.order_split_index_flag中，则代表其是无路可走了的情况
            if selected_enterprise_in_batch_i == 0 and self.cur_finished_task[batch_i] not in self.order_split_index_flag[batch_i]:
                # self.cur_time_record 不需要动
                # cur_time需要更新
                self.cur_time[batch_i] += waiting_time_piece
                # 成本不用动 (0)
                # self.cur_finished_task不用动
                # 时间就是等待时间
                # 将上述信息进行收集
                self.temp_order_time_cost[batch_i] += waiting_time_piece
                self.temp_order_expenditure[batch_i] += 0
                self.temp_order_reliability[batch_i] += 0

                # 无路可走的情况下注意将当前位置改回上一个位置
                cur_coord[batch_i][0] = self.cur_coord[batch_i][0]

            # 证明选择下一步回到depot,这里还有一个物流时间,但是这个条件下也可能存在无路可走的情况
            elif selected_enterprise_in_batch_i == 0 and self.cur_finished_task[batch_i] in self.order_split_index_flag[batch_i]:
                # 如果当前位置不是depot，说明一个order结束后该回到depot
                if not self.cur_coord[batch_i][0].equal(init_pos):

                    # 更新时间消耗
                    temp_logistic_distance = (cur_coord[batch_i][0] - self.cur_coord[batch_i][0]).norm(p=2, dim=0)  # 物流时间
                    temp_logistic_time_in_batch_i = temp_logistic_distance * logistic_time_unit

                    self.temp_order_time_cost[batch_i] += temp_logistic_time_in_batch_i

                    # 更新物流金钱消耗
                    self.temp_order_expenditure[batch_i] += temp_logistic_distance * logistic_expenditure_unit

                    # 更新当前的时间为物流时间
                    self.cur_time[batch_i] += temp_logistic_time_in_batch_i

                    # self.cur_finished_task[batch_i] += torch.tensor(1)

                    # 更新上述信息
                    # Time
                    self.batch_time_cost[batch_i].append(self.temp_order_time_cost[batch_i])
                    self.temp_order_time_cost[batch_i] = 0
                    # Exp
                    self.batch_expenditure_cost[batch_i].append(self.temp_order_expenditure[batch_i])
                    self.temp_order_expenditure[batch_i] = 0
                    # Rel
                    self.batch_reliability_cost[batch_i].append(self.temp_order_reliability[batch_i])
                    self.temp_order_reliability[batch_i] = 0

                    # 更新当前时间为0
                    self.cur_time[batch_i] = 0

                # 如果当前位置已经是depot了，那说明下一步是无路可走了
                else:
                    self.cur_time[batch_i] += waiting_time_piece
                    # 成本不用动 (0)
                    # self.cur_finished_task不用动
                    # 时间就是等待时间
                    # 将上述信息进行收集
                    self.temp_order_time_cost[batch_i] += waiting_time_piece
                    self.temp_order_expenditure[batch_i] += 0
                    self.temp_order_reliability[batch_i] += 0

                    # 无路可走的情况下注意将当前位置改回上一个位置
                    cur_coord[batch_i][0] = self.cur_coord[batch_i][0]

            # 正常情况
            else:

                # 更新当前时间
                logistic_distance = (cur_coord[batch_i][0] - self.cur_coord[batch_i][0]).norm(p=2, dim=0)
                temp_logistic_time_in_batch_i = logistic_distance*logistic_time_unit  # 物流时间
                temp_service_time_in_batch_i = self.service_time[batch_i][selected_service_in_batch_i]/(selected_enterprise_abl_in_batch_i + kf1)  # 服务时间
                temp_time_cost_in_batch_i = temp_logistic_time_in_batch_i + temp_service_time_in_batch_i
                # if batch_i == 0:
                #     print('0 batch 的 abl,时间和调运时间：', selected_enterprise_abl_in_batch_i[0],temp_service_time_in_batch_i/(selected_enterprise_abl_in_batch_i[0] + kf1), temp_logistic_time_in_batch_i)

                # 更新服务开始和结束时间表,中间只间隔了一个服务时间
                self.cur_time_record[batch_i][selected_enterprise_in_batch_i-1].append(
                    [self.cur_time[batch_i] + temp_logistic_time_in_batch_i,
                     self.cur_time[batch_i] + temp_logistic_time_in_batch_i + temp_service_time_in_batch_i]
                )
                # 更新当前的时间为调运到服务点并进行加工的时间
                self.cur_time[batch_i] += temp_time_cost_in_batch_i

                # 更新当前的任务进展
                self.cur_finished_task[batch_i] = self.cur_finished_task[batch_i] + torch.tensor(1)

                # 计算成本
                temp_logistic_expenditure_in_batch_i = logistic_expenditure_unit*logistic_distance
                temp_service_expenditure_in_batch_i = self.service_expenditure[batch_i][selected_service_in_batch_i]*(kf2 + 0.1*selected_enterprise_abl_in_batch_i)
                temp_expenditure_cost_in_batch_i = temp_logistic_expenditure_in_batch_i + temp_service_expenditure_in_batch_i

                # 计算可靠度
                temp_reliability_cost_in_batch_i = selected_enterprise_abl_in_batch_i*kf3

                # 将上述信息进行收集
                self.temp_order_time_cost[batch_i] += temp_time_cost_in_batch_i
                self.temp_order_expenditure[batch_i] += temp_expenditure_cost_in_batch_i
                self.temp_order_reliability[batch_i] += temp_reliability_cost_in_batch_i

        prev_selected = selected

        return self._replace(prev_selected=prev_selected, cur_coord=cur_coord, i=self.i + 1)

    # 时间mask和服务mask
    def get_mask(self):
        """
         Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited

        根据的就是当前的状态(这些self变量)
        只需要满足两个条件：1、下一个地方有需要的服务。2、下一个地方在该时刻没有被占用
        此外，当一个order完成后还要回到原点

        1 代表mask  0 代表不mask

        在mask函数里不应该动time这个变量，这个属于状态更新才有的
        """
        mask = []

        for batch_i, eve in enumerate(self.order_split_index_flag):
            loc_n = self.coords.size(-2)
            temp_task_index = int(self.cur_finished_task[batch_i].item())
            # 如果这个batch已经完成了调度任务的话
            if temp_task_index == len(self.order[batch_i]) - 1:
                mask.append([0] + [1 for i in range(loc_n - 1)])
                continue

            if temp_task_index in eve and not self.cur_coord[batch_i][0].equal(init_pos):  # 证明下一步要回到depot了,注意这块条件不能太简单，还得保障他能跳出来。
                mask.append([0] + [1 for i in range(loc_n - 1)])

            else:  # 证明当前不用回到起点
                next_order = self.order[batch_i][temp_task_index + 1]  # 注意mask的是下一步的内容, 这个+1是一定不会越界的
                temp_enterprise_service = self.enterprise_services[batch_i]
                temp_mask = [1]
                for enterprise_i, eve_enterprise in enumerate(temp_enterprise_service):
                    if eve_enterprise[next_order] == 1:  # 这个公司有这个服务 eve_enterprise是一个onehot编码
                        # 还需要进一步对时间占用问题进行mask，在初步的模型中认为一个公司只有一个位置
                        # 判断当前时间与公司的占用时间是否有交集
                        logistics_time = (self.cur_coord[batch_i][0]-self.coords[batch_i][enterprise_i]).norm(p=2, dim=0)
                        temp_time_begin = self.cur_time[batch_i] + logistics_time
                        temp_time_over = temp_time_begin + self.service_time[batch_i][next_order]

                        # 判断这个时间段是否在该公司的已经被占用的时间段里，在这里注意capacity的限制
                        if self._is_interpreted_in_occupy(temp_time_begin, temp_time_over, enterprise_i, batch_i, next_order):
                            temp_mask.append(1)
                        else:
                            temp_mask.append(0)
                    else:
                        temp_mask.append(1)

                # 证明此时里面的内容全部都被mask了，并且此时还不应该回到起点
                if sum(temp_mask) == len(temp_mask):
                    temp_mask = [0] + [1 for i in range(loc_n - 1)]  # 与
                mask.append(temp_mask)
        res = torch.tensor(mask, dtype=torch.bool)[:, None, :]
        if is_use_GPU:
            res = res.cuda()
        return res

    # 判断去这个公司的话和别人是否时间冲突
    # 冲突的数量需要小于该服务的容量
    # 能进入这个函数证明一定是存在这个服务
    def _is_interpreted_in_occupy(self, temp_time_begin, temp_time_over, enterprise_i, batch_i, temp_order):
        enterprise_occupied_time = self.cur_time_record[batch_i][enterprise_i]  # [[],[]]
        interpreted_times = 0
        if len(enterprise_occupied_time) == 0:
            return False
        else:
            for eve_occupied in enterprise_occupied_time:
                # 如果时间有交叉的话
                if (temp_time_over > eve_occupied[0]) and (temp_time_begin < eve_occupied[1]):
                    interpreted_times += 1  # 有一个冲突就+1
            # 判断冲突数量是否小于容量
            if interpreted_times < self.service_capacity[batch_i][enterprise_i][temp_order]:
                return False
            else:  # 否则等于时代表已经满了，不能再选这个了
                return True

    # 终止条件应该是所有order已经完成了
    def all_finished(self):
        for batch_i, eve in enumerate(self.cur_finished_task):
            # 不仅任务完成，并且回到了初始化点
            if eve == len(self.order[batch_i])-1 and self.cur_coord[batch_i][0].equal(init_pos):
                continue
            else:
                return False
        return True

    def get_current_node(self):
        return self.prev_selected

    # 得到当前整个状态从初始化到结束的cost
    # 返回一个batch的final-cost
    def get_current_cost(self):
        assert self.all_finished()  # 首先保证整个episode结束了
        assert len(self.batch_time_cost[0]) == len(self.order_split_index_flag[0])  # order数一致
        batch_size = self.order.size(0)
        final_cost = []
        for batch_i in range(batch_size):
            temp_cost = self.get_objective(batch_i, self.batch_time_cost[batch_i], self.batch_expenditure_cost[batch_i], self.batch_reliability_cost[batch_i])
            # temp_cost = max(self.batch_time_cost[batch_i]) + sum(self.batch_expenditure_cost[batch_i]) - sum(self.batch_reliability_cost[batch_i])
            final_cost.append(temp_cost.item())
        if is_use_GPU:
            return torch.tensor(final_cost).cuda()
        else:
            # print(torch.tensor(final_cost))
            return torch.tensor(final_cost)

    # 这个函数是给DRLs算法使用的
    def get_step_cost(self):
        assert self.order.shape[0] == 1, 'There is more than one batch !'
        # res = self.get_objective(0, self.batch_time_cost[0], self.batch_expenditure_cost[0], self.batch_reliability_cost[0]) if len(self.batch_time_cost[0]) != 0 else 0
        return self.temp_order_time_cost[0], self.temp_order_expenditure[0], self.temp_order_reliability[0]

    # 这个函数是用来计算cost objective的
    # 输入是待求解的第i个batch以及三个一维的列表
    # 先按照各自可能的最大值进行归一化，然后再相加
    def get_objective(self, batch_i, time_list, expenditure_list, reliability_list):
        # print(max(time_list), sum(expenditure_list), sum(reliability_list))
        # tensor(25.3721) tensor([16571.4785]) tensor([737.7808])  目前这三个cost不在一个数量级上
        return (W_t * max(time_list)/self.max_time[batch_i] + W_e * sum(expenditure_list)/self.max_expenditure[batch_i]) - (W_r * sum(reliability_list)/self.max_reliability[batch_i])

    def get_is_next_processing(self):
        batch_size = self.cur_coord.shape[0]
        # self.next_is_processing = torch.ones(batch_size, dtype=torch.bool)  # 默认全是true
        for batch_i in range(batch_size):
            # 证明这个batch已经结束了调度 or 下一步要回到depot了；则证明下一步不是processing
            if self.cur_finished_task[batch_i] == len(self.order[batch_i]) - 1 or self.cur_finished_task[batch_i] in self.order_split_index_flag[batch_i]:
                self.next_is_processing[batch_i] = False
            else:
                self.next_is_processing[batch_i] = True
        return self.next_is_processing

    # 这个函数用来获得最后时刻三个优化目标各自的值
    def get_detailed_current_cost(self):
        assert self.all_finished()  # 首先保证整个episode结束了
        assert len(self.batch_time_cost[0]) == len(self.order_split_index_flag[0])  # order数一致
        batch_size = self.order.size(0)
        final_detailed_costs = []
        for batch_i in range(batch_size):
            t1,t2,t3,t4,t5,t6 = self.get_detailed_objective(batch_i, self.batch_time_cost[batch_i], self.batch_expenditure_cost[batch_i], self.batch_reliability_cost[batch_i])
            # temp_cost = max(self.batch_time_cost[batch_i]) + sum(self.batch_expenditure_cost[batch_i]) - sum(self.batch_reliability_cost[batch_i])
            final_detailed_costs.append([t1.item(), t2.item(), t3.item(), t4.item(), t5.item(), t6.item()])
        return final_detailed_costs

    # 也是用于获取三个优化目标的值
    def get_detailed_objective(self, batch_i, time_list, expenditure_list, reliability_list):
        # print(max(time_list), sum(expenditure_list), sum(reliability_list))
        # tensor(25.3721) tensor([16571.4785]) tensor([737.7808])  目前这三个cost不在一个数量级上
        return max(time_list)/self.max_time[batch_i], max(time_list), sum(expenditure_list)/self.max_expenditure[batch_i], sum(expenditure_list), sum(reliability_list)/self.max_reliability[batch_i],sum(reliability_list)
