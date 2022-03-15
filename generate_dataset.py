import json
import os
import random
import torch
from param_parser import parameter_parser
from torch_geometric.data import Dataset, Data
import scipy.stats as stats


def initialize_embedding(size, device):
    """
    Generate initial embedding of players.(服从均值为0.5，标准差为1，分布在[0,1]的截断正态分布)
    :param size: Size of embedding.
    :param device: Device of tensor.
    :return: Initialized embedding.
    """
    mu, sigma = 0.5, 1
    lower, upper = 0, 1
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 生成分布
    h = X.rvs(size)  # 取样
    return torch.tensor(h, dtype=torch.float, device=device)

def generate_empty_graph(player_num, config):
    """
    Generate an empty graph to pad batch
    :param player_num: Number of players in the game
    :param config: Model configuration
    :return: An empty graph with no edges
    """
    data = Data()
    data.x = torch.zeros((player_num, config.embedding_size), device=config.device)
    data.y = torch.unsqueeze(torch.zeros(10, device=config.device, dtype=torch.long), dim=0)
    data.L_pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
    data.G_pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
    data.U_pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
    data.L_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
    data.G_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
    data.U_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
    data.player_num = player_num
    data.extra_info = torch.zeros((player_num, 2), device=config.device)

    return data


def generate_graph(record, config):
    """
    生成以图形式表示的数据集
    :param config:
    :param record:简化后的json记录
    :return: data_list 该对局本身的图和填充的图构成的列表
    """
    data_list = []
    label = torch.tensor(record['rolesTensor'], dtype=torch.long, device=config.device)  # 真实身份
    game = record['gameProcess']  # 当前游戏
    mission_history = record['missionHistory']  # 任务历史
    group_history = record['groupHistory']  # 组队历史
    player_num = record["numberOfPlayers"]  # 玩家人数
    mask = []  # 记录哪些图是填充的
    mission_cnt = 0
    for mission in game:  # 当前任务
        add_info = []  # 在LSTM中添加的额外信息
        for _ in range(player_num):
            add_info.append(torch.zeros(2, device=config.device))  # 第一维：玩家是否在本轮任务中参与组队 第二维：本轮任务是否成功
        vote_cnt = 0
        for vote in mission:  # 当前投票
            mask.append(1)
            role = vote[0]  # 玩家角色
            voteResult = vote[1]  # 投票结果

            data = Data()
            data.x = initialize_embedding((player_num, config.embedding_size), device=config.device)
            data.y = torch.unsqueeze(label, dim=0)
            data.L_pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
            data.G_pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
            data.U_pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
            data.L_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
            data.G_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
            data.U_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=config.device)
            data.player_num = player_num

            # data.extra_info = []
            # for _ in range(player_num):
            #     data.extra_info.append(torch.zeros(2, device=config.device))

            data.extra_info = torch.zeros((player_num, 2), device=config.device)

            # 添加任务成功与否信息
            mission_result = mission_history[mission_cnt]
            group_result = group_history[mission_cnt]
            for i in range(player_num):
                info = data.extra_info[i]
                info[0] = group_result[i]
                info[1] = mission_result

            for u in range(player_num):
                for v in range(player_num):
                    if u == v:
                        continue
                    # 正边
                    if voteResult[u] == voteResult[v]:
                        if role[u] == 'Member':
                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=config.device)
                            data.G_pos_edge_index = torch.cat((data.G_pos_edge_index, new_edge), dim=-1)
                        elif role[u] == 'nonMember':
                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=config.device)
                            data.U_pos_edge_index = torch.cat((data.U_pos_edge_index, new_edge), dim=-1)
                        else:
                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=config.device)
                            # if data.L_pos_edge_index[0][0] == -1:
                            #     data.L_pos_edge_index = new_edge
                            data.L_pos_edge_index = torch.cat((data.L_pos_edge_index, new_edge), dim=-1)
                    # 负边
                    else:
                        if role[u] == 'Member':
                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=config.device)
                            data.G_neg_edge_index = torch.cat((data.G_neg_edge_index, new_edge), dim=-1)
                        elif role[u] == 'nonMember':
                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=config.device)
                            data.U_neg_edge_index = torch.cat((data.U_neg_edge_index, new_edge), dim=-1)
                        else:
                            new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=config.device)
                            data.L_neg_edge_index = torch.cat((data.L_neg_edge_index, new_edge), dim=-1)
            data_list.append(data)
            vote_cnt += 1

        # 填充
        empty_graph = generate_empty_graph(player_num, config)
        while vote_cnt < 5:
            mask.append(0)
            data_list.append(empty_graph)
            vote_cnt += 1
        mission_cnt += 1
    # 暂时的解决方案
    while len(data_list) < 25:
        mask.append(0)
        data_list.append(empty_graph)

    mask = torch.unsqueeze(torch.tensor(mask, device=config.device), dim=0)
    for data in data_list:
        data.mask = mask

    return data_list


class AvalonDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.config = parameter_parser()
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):  # 不需要包含绝对路径，只用文件名
        return [str(i) + '.json' for i in range(1, self.config.dataset_size)]

    @property
    def processed_file_names(self):
        return ['data_' + str(i) + '.pt' for i in range(0, self.config.dataset_size*25)]  # 每张图同一扩充至25

    def download(self):
        pass

    def process(self):
        idx = 0
        print(self.config.dataset_size)
        for ROOT, dirs, files in os.walk(self.raw_dir):  # raw_dir是预设属性，默认为root+‘/raw’
            random.seed(1234)
            random.shuffle(list(files))
            for file in files:
                filename = self.raw_dir + '\\' + str(file)
                with open(filename, 'r', encoding="utf8") as f:
                    gameData = json.loads(f.read())
                data_list = generate_graph(gameData, config=self.config)
                for data in data_list:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                    idx += 1
                # print(idx)
                # # 先尝试建立一个小数据集
                # if idx == 2500:
                #     break

        # 不需要将数据集物理分割，对列表切片即可
        # train_dataset = dataset[:800000]
        # val_dataset = dataset[800000:900000]
        # test_dataset = dataset[900000:]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
