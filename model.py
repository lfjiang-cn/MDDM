import math
import torch
import torch.nn as nn
from torch.nn import init, Softmax
import torch.nn.functional as F
from loss import MMDLoss
from utils import weights_init_classifier, weights_init_kaiming
from resnet import resnet50
from torch.cuda.amp import autocast
import torch.fft as fft
from timm.models.layers import DropPath

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Shift(nn.Module):
    def __init__(self, mode='max'):
        super(Shift, self).__init__()

        self.mode = mode
        if self.mode == 'max':
            self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
            self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        elif self.mode == 'avg':
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        else:
            raise ValueError("Invalid mode! mode must be 'max' or 'avg'.")

    def forward(self, x):
        bias_h = self.pool_h(x)
        bias_h = F.softmax(bias_h, dim=2)
        bias_w = self.pool_w(x)
        bias_w = F.softmax(bias_w, dim=3)
        bias = torch.matmul(bias_h, bias_w)

        return bias

class lsk(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3) # 应用padding使输入输出shape保持一致
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)


    def forward(self, x):
        # x: (B,C,H,W)
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)

        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True) #
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv_m(attn)
        return x * attn


class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x ** p, dim=-1) + eps) ** (1 / p)


def pha_unwrapping(x):
    fft_x = torch.fft.fft2(x.clone(), dim=(-2, -1))
    fft_x = torch.stack((fft_x.real, fft_x.imag), dim=-1)
    pha_x = torch.atan2(fft_x[:, :, :, :, 1], fft_x[:, :, :, :, 0])

    fft_clone = torch.zeros(fft_x.size(), dtype=torch.float).cuda()
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x.clone())
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x.clone())

    # get the recomposed image: source content, target style
    pha_unwrap = torch.fft.ifft2(torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
                                 dim=(-2, -1)).float()

    return pha_unwrap



# DLF
class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class DLF(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(DLF, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z

#################################################################
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SE_Block(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Adaptive_Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., reduction=4):
        super().__init__()
        self.dwconv0 = Conv(dim, dim, 7, g=dim, act=False)
        self.dwconv1 = Conv(dim, dim, 5, g=dim, act=False)
        self.se = SE_Block(dim, reduction)  # Add SE module for channel attention
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.hidden_weight = nn.Parameter(torch.ones(dim, 1, 1))  # learnable weight for high-dimensional control

    def forward(self, x):
        input = x
        x = self.dwconv0(x) + self.dwconv1(x) + x
        x = self.se(x)  # Apply SE block
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = x * self.hidden_weight  # Apply learnable weight for implicit high-dimensional control
        x = input + self.drop_path(x)
        return x


###############################################################

class visible_module(nn.Module):  # block 1 2 权重不共享
    def __init__(self, pretrained=True):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v
        self.visible.layer3 = None
        self.visible.layer4 = None

        self.DLF1 = Adaptive_Star_Block(256)
        self.DLF2 = Adaptive_Star_Block(512)

    def forward(self, x):
        x = x + 0.8 * pha_unwrapping(x)
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        x_ = x
        x = self.visible.layer1(x_)
        x_ = self.DLF1(x)
        x = self.visible.layer2(x_)
        x = self.DLF2(x)
        return x


class thermal_module(nn.Module):  # block 1 2 权重不共享
    def __init__(self, pretrained=True):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t
        self.thermal.layer3 = None
        self.thermal.layer4 = None

        self.DLF1 = Adaptive_Star_Block(256)
        self.DLF2 = Adaptive_Star_Block(512)

    def forward(self, x):
        x = x + 0.8 * pha_unwrapping(x)
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)

        x_ = x
        x = self.thermal.layer1(x_)
        x_ = self.DLF1(x)
        x = self.thermal.layer2(x_)
        x = self.DLF2(x)
        return x





class base_module(nn.Module):  # block 3 4 权重共享
    def __init__(self, pretrained=True):
        super(base_module, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.base = base
        self.base.conv1 = None
        self.base.bn1 = None
        self.base.relu = None
        self.base.maxpool = None
        self.layer1 = None
        self.layer2 = None

        self.block1 = Adaptive_Star_Block(1024)
        self.block2 = Adaptive_Star_Block(2048)

        self.shift1 = Shift(mode='max')
        self.shift2 = Shift(mode='max')

    def forward(self, x):
        x = self.base.layer3(x)
        # MIA
        x = self.block1(x)
        x = self.base.layer4(x)
        x = self.block2(x)

        return x


class HSICLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(HSICLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, x, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(x.size(0))
        total0 = x.unsqueeze(0).expand(n_samples, n_samples, x.size(1))
        total1 = x.unsqueeze(1).expand(n_samples, n_samples, x.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # 合并 source 和 target
        X = torch.cat([source, target], dim=0)   # [n+m, d]
        n_samples = X.size(0)

        # 构建 domain label（source=0, target=1）
        domain_label = torch.cat([
            torch.zeros(source.size(0), 1).to(source.device),
            torch.ones(target.size(0), 1).to(target.device)
        ], dim=0)

        # 计算核矩阵
        K = self.guassian_kernel(X, self.kernel_mul, self.kernel_num, self.fix_sigma)
        L = self.guassian_kernel(domain_label, self.kernel_mul, self.kernel_num, self.fix_sigma)

        # 中心化矩阵 H
        H = torch.eye(n_samples).to(X.device) - (1.0 / n_samples) * torch.ones((n_samples, n_samples)).to(X.device)

        # HSIC = trace(KHLH) / (n-1)^2
        KH = torch.matmul(K, H)
        LH = torch.matmul(L, H)
        hsic = torch.trace(torch.matmul(KH, LH)) / ((n_samples - 1) ** 2)

        return hsic



# Hilbert space matching loss
class HSM(nn.Module):
    def __init__(self, ):
        super(HSM, self).__init__()
        self.HSIC = HSICLoss(max_moment=5)

    def forward(self, x, x_hat):
        b, c = x.shape
        inter = self.HSIC(F.normalize(x[:b // 2], p=2, dim=1),
                         F.normalize(x[b // 2:], p=2, dim=1)) \
              + self.HSIC(F.normalize(x_hat[:b // 2], p=2, dim=1),
                         F.normalize(x_hat[b // 2:], p=2, dim=1))

        intra = self.HSIC(F.normalize(x[:b // 2], p=2, dim=1),
                         F.normalize(x_hat[:b // 2], p=2, dim=1)) \
              + self.HSIC(F.normalize(x[b // 2:], p=2, dim=1),
                         F.normalize(x_hat[b // 2:], p=2, dim=1))

        hsm_loss = 0.45 * inter + 0.05 * intra
        return hsm_loss


# Euclidean space matching loss
class ESM(nn.Module):
    def __init__(self, class_num, pool_dim=2048, tau=0.2):
        super(ESM, self).__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.ID = nn.CrossEntropyLoss()
        self.tau = tau
        self.visible_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.infrared_classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.visible_classifier_ = nn.Linear(pool_dim, class_num, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data

        self.infrared_classifier_ = nn.Linear(pool_dim, class_num, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

    def forward(self, x, x_hat, label1, label2):
        b, c = x.shape
        x_v = self.visible_classifier(x[:b // 2])
        x_t = self.infrared_classifier(x[b // 2:])
        x_hat_v = self.visible_classifier(x_hat[:b // 2])
        x_hat_t = self.infrared_classifier(x_hat[b // 2:])

        logit_x_id = [x_v, x_t]
        logit_x_hat_id = [x_hat_v, x_hat_t]

        logit_x = torch.cat((x_v, x_t), 0).float()
        logit_x_hat = torch.cat((x_hat_v, x_hat_t), 0).float()

        with torch.no_grad():
            # update the W
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.tau) \
                                                    + self.infrared_classifier.weight.data * self.tau
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.tau) \
                                                   + self.visible_classifier.weight.data * self.tau

            logit_a_v = self.infrared_classifier_(x[:b // 2])
            logit_a_t = self.visible_classifier_(x[b // 2:])
            logit_a_hat_v = self.infrared_classifier_(x_hat[:b // 2])
            logit_a_hat_t = self.visible_classifier_(x_hat[b // 2:])

            logit_a = torch.cat((logit_a_v, logit_a_t), 0).float()
            logit_a_hat = torch.cat((logit_a_hat_v, logit_a_hat_t), 0).float()

        # student
        logit_x = F.softmax(logit_x, 1)
        logit_x_hat = F.softmax(logit_x_hat, 1)
        # teacher
        logit_a = F.softmax(logit_a, 1)
        logit_a_hat = F.softmax(logit_a_hat, 1)

        kl1 = self.KLDivLoss(logit_a.log(), logit_x) + self.KLDivLoss(logit_x.log(), logit_a)
        kl2 = self.KLDivLoss(logit_a_hat.log(), logit_x_hat) + self.KLDivLoss(logit_x_hat.log(), logit_a_hat)

        ce1 = 0.25 * (self.ID(logit_x_id[0], label1) + self.ID(logit_x_id[1], label2))
        ce2 = 0.25 * (self.ID(logit_x_hat_id[0], label1) + self.ID(logit_x_hat_id[1], label2))

        esm_loss = kl1 + kl2 + ce1 + ce2
        return esm_loss

class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()

        self.visible = visible_module(pretrained=pretrained)
        self.thermal = thermal_module(pretrained=pretrained)
        self.base = base_module(pretrained=pretrained)
        #self.CSS = CSS()
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.HSM = HSM()
        self.ESM = ESM(class_num=class_num, pool_dim=pool_dim)

    @autocast()
    def forward(self, x_v, x_t, label1=None, label2=None, modal=0):
        if modal == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), 0)
            del x_v, x_t

        elif modal == 1:
            x = self.visible(x_v)

        elif modal == 2:
            x = self.thermal(x_t)

        if self.training:
            x = self.base(x)
            b, c, h, w = x.shape
            x_hat = self.gap(x)
            x_hat = x_hat.view(b, -1)

            # all value must > 0 due to the GeMP
            x = self.relu(x)
            x = x.view(b, c, h * w)
            x = self.pool(x)

            hsm_loss = self.HSM(x, x_hat)

            x_after_BN = self.bottleneck(x)

            cls_id = self.classifier(x_after_BN)
            esm_loss = self.ESM(x_after_BN, x_hat, label1, label2)

            return {
                'cls_id': cls_id,
                'feat': x_after_BN,
                'hsm': hsm_loss,
                'esm': esm_loss,
            }

        else:
            x = self.base(x)
            # in the testing phase, the CSS is removed.
            b, c, h, w = x.shape
            x = self.relu(x)
            x = x.view(b, c, h * w)
            x = self.pool(x)
            x_after_BN = self.bottleneck(x)

        return F.normalize(x, p=2.0, dim=1), F.normalize(x_after_BN, p=2.0, dim=1)




