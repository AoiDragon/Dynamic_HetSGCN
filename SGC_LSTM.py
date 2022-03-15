import torch
from hetero_signed_conv import HeteroSignedConv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import ReLU


class SGC_LSTM(torch.nn.Module):
    """
    SGC_LSTM NetWork Class.
    """

    def __init__(self, device, config):
        """
        Initialize SGC_LSTM.
        :param device: Device for calculations.
        :param config: Arguments object.
        """
        super(SGC_LSTM, self).__init__()
        self.device = device
        self.config = config
        self.in_channels = config.sgcn_input_size
        self.hidden_channels = config.sgcn_hidden_size
        self.out_channels = config.sgcn_output_size
        self.setup_layers()

    def setup_layers(self):
        self.first_aggr = HeteroSignedConv(self.in_channels, self.hidden_channels, first_aggr=True)
        self.aggr = HeteroSignedConv(self.hidden_channels, self.out_channels, first_aggr=False)
        self.lstm = torch.nn.LSTM(self.config.lstm_input_size, self.config.lstm_hidden_size, batch_first=True)
        self.W = torch.nn.Linear(self.config.lstm_hidden_size, self.config.embedding_size)  # bias默认为True
        self.relu = ReLU()

    def forward(self, batch):
        """
        :param batch: Batch data of graphs
        :return: out: List of prediction for every graph in the batch
        """
        # HeteroSGCN
        out = []
        extra_info = batch['extra_info']
        h = self.first_aggr(batch)
        h = self.relu(self.aggr(h))

        # Get the list of player_num
        player_num_list = []
        L = batch['player_num'].size()[0]
        for i in range(0, L, 25):
            player_num_list.append(batch['player_num'][i])

        # Resize tensor
        game_num = len(player_num_list)
        start = 0
        mask_start = 0

        # batch_embedding = torch.zeros((0, 25, 256), device=self.config.device)
        batch_embedding = []
        # batch_extra_info = torch.zeros((0, 25, 2), device=self.config.device)
        batch_mask = batch['mask']
        length = []
        for game in range(game_num):
            player_num = player_num_list[game]
            # game_embedding = torch.zeros((player_num, 0), device=self.config.device)

            end = start + player_num * 25
            # game_embedding = h[start:end, :].view(25, player_num, 256)
            game_embedding = h[start:end, :].view(25, player_num, 2*self.config.sgcn_output_size)
            game_extra_info = extra_info[start:end, :].view(25, player_num, 2)
            game_mask = batch_mask[mask_start]
            for i in range(player_num):
                length.append(int(sum(game_mask)))
            game_mask = torch.squeeze(torch.nonzero(game_mask).t())

            # 交换第二和第三个维度，变成player_num, 25, 256
            game_embedding = game_embedding.transpose(dim0=0, dim1=1)
            game_extra_info = game_extra_info.transpose(dim0=0, dim1=1)
            game_embedding = torch.cat((game_embedding, game_extra_info), dim=-1)

            #
            game_embedding = torch.index_select(game_embedding, 1, game_mask)

            # batch_embedding = torch.cat((batch_embedding, game_embedding), dim=0)
            for i in range(game_embedding.size()[0]):
                batch_embedding.append(game_embedding[i])
            # batch_extra_info = torch.cat((batch_extra_info, game_extra_info), dim=0)

            start = end
            mask_start += 25

        # batch_embedding = torch.cat((batch_embedding, batch_extra_info), dim=-1)
        batch_embedding = pad_sequence(batch_embedding, False).transpose(dim0=0, dim1=1)
        # batch_embedding为sum(player_num) * 25 * (256+2)

        h0 = torch.randn(1, batch_embedding.size()[0], self.config.lstm_hidden_size, device=self.config.device)
        c0 = torch.randn(1, batch_embedding.size()[0], self.config.lstm_hidden_size, device=self.config.device)

        batch_embedding = pack_padded_sequence(batch_embedding, length, True, False)

        _, (h_last, _) = self.lstm(batch_embedding, (h0, c0))  # h_last: sum(player_num)*hidden_size

        h_last = h_last[0]
        h_last = self.W(h_last)  # h_last:sum(player_num)*hidden_size

        # Divide tensor into different games
        start = 0
        for game in range(game_num):
            end = start + player_num_list[game]
            h_now = h_last[start:end, :]
            out.append(h_now)
            start = end

        return out

