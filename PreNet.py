# --coding:utf-8--
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class PreNet(nn.Module):
    def __init__(self):
        super(PreNet, self).__init__()
        self.net = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        loadpath = ""
        checkpoint = torch.load(loadpath)  # 加载模型到self.dev设备上
        self.net.load_state_dict(checkpoint["net_state"], strict=True)
        print("loading")

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == "__main__":
    model = ZYHNet()
    indata = torch.ones(3, 3, 224, 224)
    output = model(indata)
    print(output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
