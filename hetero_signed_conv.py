import torch
from torch_geometric.typing import PairTensor, Adj
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class HeteroSignedConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):
        """
        :param in_channels: size of input node feature
        :param out_channels: size of output node feature
        :param first_aggr: whether the aggregator is the first one or not
        """
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos_l = Linear(in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
        else:
            self.lin_pos_l = Linear(6 * in_channels, out_channels, False)  # 三种角色，两种嵌入，拼接后为六种
            self.lin_pos_r = Linear(3 * in_channels, out_channels, bias)   # 在第一层中pos_embedding占三个
            self.lin_neg_l = Linear(6 * in_channels, out_channels, False)
            self.lin_neg_r = Linear(3 * in_channels, out_channels, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()

    def forward(self, batch):
        x = batch['x']
        y = batch['y']
        L_pos_edge_index = batch['L_pos_edge_index']
        G_pos_edge_index = batch['G_pos_edge_index']
        U_pos_edge_index = batch['U_pos_edge_index']
        L_neg_edge_index = batch['L_neg_edge_index']
        G_neg_edge_index = batch['G_neg_edge_index']
        U_neg_edge_index = batch['U_neg_edge_index']

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.first_aggr:
            out_L_pos = self.lin_pos_l(self.propagate(L_pos_edge_index, x=x, size=None))
            out_G_pos = self.lin_pos_l(self.propagate(G_pos_edge_index, x=x, size=None))
            out_U_pos = self.lin_pos_l(self.propagate(U_pos_edge_index, x=x, size=None))
            out_L_pos += self.lin_pos_r(x[1])
            out_G_pos += self.lin_pos_r(x[1])
            out_U_pos += self.lin_pos_r(x[1])

            out_L_neg = self.lin_neg_l(self.propagate(L_neg_edge_index, x=x, size=None))
            out_G_neg = self.lin_neg_l(self.propagate(G_neg_edge_index, x=x, size=None))
            out_U_neg = self.lin_neg_l(self.propagate(U_neg_edge_index, x=x, size=None))
            out_L_neg += self.lin_neg_r(x[1])
            out_G_neg += self.lin_neg_r(x[1])
            out_U_neg += self.lin_neg_r(x[1])

            data = Data()
            data.x = torch.cat([out_L_pos, out_G_pos, out_U_pos, out_L_neg, out_G_neg, out_U_neg], dim=-1)
            data.y = y
            data.L_pos_edge_index = L_pos_edge_index
            data.G_pos_edge_index = G_pos_edge_index
            data.U_pos_edge_index = U_pos_edge_index
            data.L_neg_edge_index = L_pos_edge_index
            data.G_neg_edge_index = G_pos_edge_index
            data.U_neg_edge_index = U_pos_edge_index
            return data

        else:
            F_in = self.in_channels
            # 沿正路径聚合正Leader邻居
            out_L_pos_Balanced = self.propagate(L_pos_edge_index, size=None,
                                                x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_G_pos_Balanced = self.propagate(G_pos_edge_index, size=None,
                                                x=(x[0][..., F_in:2*F_in], x[1][..., F_in:2*F_in]))
            out_U_pos_Balanced = self.propagate(U_pos_edge_index, size=None,
                                                x=(x[0][..., 2*F_in:3*F_in], x[1][..., 2*F_in:3*F_in]))
            out_L_pos_Unbalanced = self.propagate(L_neg_edge_index, size=None,
                                                  x=(x[0][..., 3*F_in:4*F_in], x[1][..., 3*F_in:4*F_in]))
            out_G_pos_Unbalanced = self.propagate(G_neg_edge_index, size=None,
                                                  x=(x[0][..., 4*F_in:5*F_in], x[1][..., 4*F_in:5*F_in]))
            out_U_pos_Unbalanced = self.propagate(U_neg_edge_index, size=None,
                                                  x=(x[0][..., 5*F_in:6*F_in], x[1][..., 5*F_in:6*F_in]))
            out_pos = torch.cat([out_L_pos_Balanced, out_G_pos_Balanced, out_U_pos_Balanced,
                                 out_L_pos_Unbalanced, out_G_pos_Unbalanced, out_U_pos_Unbalanced], dim=-1)
            out_pos = self.lin_pos_l(out_pos)
            # 给每个点加上自己的pos_embedding
            out_pos += self.lin_pos_r(x[1][..., :3*F_in])

            out_L_neg_Balanced = self.propagate(L_neg_edge_index, size=None,
                                                x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_G_neg_Balanced = self.propagate(G_neg_edge_index, size=None,
                                                x=(x[0][..., F_in:2 * F_in], x[1][..., F_in:2 * F_in]))
            out_U_neg_Balanced = self.propagate(U_neg_edge_index, size=None,
                                                x=(x[0][..., 2 * F_in:3 * F_in], x[1][..., 2 * F_in:3 * F_in]))
            out_L_neg_Unbalanced = self.propagate(L_pos_edge_index, size=None,
                                                  x=(x[0][..., 3 * F_in:4 * F_in], x[1][..., 3 * F_in:4 * F_in]))
            out_G_neg_Unbalanced = self.propagate(G_pos_edge_index, size=None,
                                                  x=(x[0][..., 4 * F_in:5 * F_in], x[1][..., 4 * F_in:5 * F_in]))
            out_U_neg_Unbalanced = self.propagate(U_pos_edge_index, size=None,
                                                  x=(x[0][..., 5 * F_in:6 * F_in], x[1][..., 5 * F_in:6 * F_in]))
            out_neg = torch.cat([out_L_neg_Balanced, out_G_neg_Balanced, out_U_neg_Balanced,
                                 out_L_neg_Unbalanced, out_G_neg_Unbalanced, out_U_neg_Unbalanced], dim=-1)
            out_neg = self.lin_neg_l(out_neg)
            # 给每个点加上自己的neg_embedding
            out_neg += self.lin_neg_r(x[1][..., 3 * F_in:])
            return torch.cat([out_pos, out_neg], dim=-1)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr})')
