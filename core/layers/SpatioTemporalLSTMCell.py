import torch
import torch.nn as nn


class self_attention_memory_module(nn.Module):  # SAM
    def __init__(self, input_dim, hidden_dim):
        super(self_attention_memory_module, self).__init__()
        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W
        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        # memory m attention #
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        # 논문의 수식과 같습니다(figure)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, hidden_dim, configs):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.input_channels = in_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = configs.filter_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.device = configs.device
        self.width = configs.img_width // configs.patch_size // configs.sr_size
        self.height = configs.img_height // configs.patch_size // configs.sr_size

        self._forget_bias = 1.0

        self.attention_layer = self_attention_memory_module(input_dim=hidden_dim, hidden_dim=hidden_dim)

        if configs.layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, hidden_dim * 7, kernel_size=self.kernel_size, padding=self.padding, bias=False),
                nn.LayerNorm([hidden_dim * 7, self.height, self.width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=self.kernel_size, padding=self.padding, bias=False),
                nn.LayerNorm([hidden_dim * 4, self.height, self.width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=self.kernel_size, padding=self.padding, bias=False),
                nn.LayerNorm([hidden_dim * 3, self.height, self.width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=False),
                nn.LayerNorm([hidden_dim, self.height, self.width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, hidden_dim * 7, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t,g_t):
        x_concat = self.conv_x(x_t)  # hidden 扩展了7倍
        h_concat = self.conv_h(h_t)  # hidden 扩展了4倍
        m_concat = self.conv_m(m_t)  # hidden 扩展了3倍
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.hidden_dim, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_dim, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        h_new, g_next = self.attention_layer(h_new, g_t)



        return h_new, c_new, m_new,g_next


if __name__ == '__main__':
    from configs.radar_train_configs import configs

    parse = configs()
    configs = parse.parse_args()
    print(configs.num_hidden)
    model = SpatioTemporalLSTMCell(64, configs.num_hidden, configs).cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
