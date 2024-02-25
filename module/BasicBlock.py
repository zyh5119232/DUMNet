
from torch import nn
# from backbone.Acon import MetaAconC
from module.weight_init import weight_init
from functools import partial
import torch
# from .drop import DropPath

# from (Robust CNN) change norm_layer and act_layer . remove drop_path
class NewConv(nn.Module):
    def __init__(self, indim, dim, drop_path=0., layer_scale_init_value=1e-6, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, act_layer=partial(nn.GELU, inplace=True),
                 stride=1, downsample=None,
                 kernel_size=11, padding=5):
        super(NewConv, self).__init__()
        self.conv_dw = nn.Conv2d(indim, dim, kernel_size=kernel_size, padding=padding, groups=indim, stride=stride, bias=True)  # depthwise conv
        self.norm1 = norm_layer(dim)
        self.pwconv1 = nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=1, bias=True)
        self.act2 = act_layer()
        self.pwconv2 = nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        # x = self.drop_path(x) + shortcut
        x = x + shortcut
        return x

    def initialize(self):
        weight_init(self)

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # offset的shape为(1,1,2*N,w,h) # 其中N=k_size*k_size; w,h是input_fea的宽高
        # 对offset的理解：offset保存的是，卷积核在input_fea上滑动时，以每个像素点为中心点时，所要聚合的邻居点的坐标索引修正值，这也是为什么每个像素点对应有2*N个通道（前N为x坐标，后N为y坐标）
        offset = self.p_conv(x)
        # print("offset:", offset.size())
        # 在卷积的乘加操作里，引入额外的权重（默认不用）
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # 获取滑动卷积的过程中，每次卷积的卷积核中心点的邻接点的索引(叠加了offset之后)
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # ============= 双线性插值计算浮点数坐标处的像素值 ===============
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 这里的x_offset的含义是，卷积核在inp_fea上滑动卷积时，以inp每个点为中心点时，卷积核各点对应的像素的像素值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        # print(x_offset.size())
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 卷积核中每个点相对于卷积核中心点的偏移
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # 卷积核在特征图上滑动的中心点
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


if __name__ == "__main__":
    device = torch.device('cuda')
    conv = DeformConv2d(inc=3, outc=10, kernel_size=3, padding=1, stride=1, bias=None, modulation=True)
    inp = torch.tensor([[128., 126., 180.] * 16]).view((1, 4, 4, 3))
    inp = inp.permute(0, 3, 1, 2).contiguous()
    print(inp.size())
    conv = conv.to(device)
    out = conv(inp.to(device))
    print(out.size())

    pass

class BasicConv2d(nn.Module):
    def __init__(self, inC, outC, kernel, stride=1, padding=1, dilation=1, groups=1, bias=False, bn_use=True, active="ReLU"):
        super(BasicConv2d, self).__init__()

        if active not in ["ReLU", "Leaky", "ACON", "None", "PReLU","Gelu"]:
            raise NotImplementedError("The argument 'active' in BasicConv2d is False")

        self.conv = nn.Conv2d(in_channels=inC,
                              out_channels=outC,
                              kernel_size=kernel,
                              stride=(stride, stride),
                              padding=padding,
                              dilation=(dilation, dilation),
                              groups=groups,
                              bias=bias)

        self.bn_use = bn_use
        if self.bn_use:
            self.bn = nn.BatchNorm2d(outC)

        self.active = active
        if self.active == "ReLU":
            self.activeLayer = nn.ReLU(inplace=True)
        elif self.active == "Leaky":
            self.activeLayer = nn.LeakyReLU(inplace=True)
        # elif self.active == "ACON":
        #     self.activeLayer = MetaAconC(width=outC)
        elif self.active == "PReLU":
            self.activeLayer = nn.PReLU()
        elif self.active == "GeLU":
            self.activeLayer = nn.GELU()

    def forward(self, x):

        out = self.conv(x)
        if self.bn_use:
            out = self.bn(out)
        if self.active != "None":
            out = self.activeLayer(out)

        return out
    def initialize(self):
        weight_init(self)

class ChannelChange(nn.Module):
    def __init__(self, inC, outC, kernel, padding, active="ReLU"):
        super(ChannelChange, self).__init__()

        self.Conv = nn.ModuleList()
        for inc, outc in zip(inC, outC):
            self.Conv.append(BasicConv2d(inC=inc, outC=outc, kernel=kernel, stride=1, padding=padding, active=active))

    def forward(self, x):

        feature = []
        for i in range(len(x)):
            feature.append(self.Conv[i](x[i]))

        return feature
