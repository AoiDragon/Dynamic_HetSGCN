import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    # Data Loading
    parser.add_argument('--dataset_size',
                        type=int,
                        default=29603,
                        # default=100,
                        help='数据集大小')

    # Model Configuration
    parser.add_argument('--embedding_size',
                        type=int,
                        default=6,
                        help="嵌入的默认长度，以标准角色配置中的数目相同")

    parser.add_argument('--layer_num',
                        type=int,
                        default=2,
                        help="sgcn层数")

    parser.add_argument('--lstm_input_size',
                        type=int,
                        default=128+2,
                        help="lstm的输入大小，默认为256（SGCN正嵌入128+SGCN负嵌入128）,暂未包括额外信息的2")

    parser.add_argument('--lstm_hidden_size',
                        type=int,
                        default=128,
                        help="lstm的隐藏层大小，默认为128")

    parser.add_argument('--sgcn_input_size',
                        type=int,
                        default=6,
                        help="sgcn的输入大小，默认为6")

    parser.add_argument('--sgcn_hidden_size',
                        type=int,
                        default=64,
                        help="sgcn的隐藏层大小，默认为32")

    parser.add_argument('--sgcn_output_size',
                        type=int,
                        default=64,
                        help="sgcn的输出大小，默认为128")

    # Training Configuration
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64)

    parser.add_argument('--epoch_num',
                        type=int,
                        default=200)

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001)

    # return parser.parse_args()
    return parser.parse_args(args=[])
