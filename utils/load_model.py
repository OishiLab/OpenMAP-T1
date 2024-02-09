import os
import torch
from utils.network import UNet

def load_model(opt, device):
    cnet = UNet(1, 1)
    cnet.load_state_dict(torch.load(os.path.join(opt.m, "CNet/CNet.pth")))
    cnet.to(device)
    cnet.eval()

    ssnet = UNet(1, 1)
    ssnet.load_state_dict(torch.load(os.path.join(opt.m, "SSNet/SSNet.pth")))
    ssnet.to(device)
    ssnet.eval()

    pnet_c = UNet(3, 142)
    pnet_c.load_state_dict(torch.load(os.path.join(opt.m, "PNet/coronal.pth")))
    pnet_c.to(device)
    pnet_c.eval()

    pnet_s = UNet(3, 142)
    pnet_s.load_state_dict(torch.load(os.path.join(opt.m, "PNet/sagittal.pth")))
    pnet_s.to(device)
    pnet_s.eval()

    pnet_a = UNet(3, 142)
    pnet_a.load_state_dict(torch.load(os.path.join(opt.m, "PNet/axial.pth")))
    pnet_a.to(device)
    pnet_a.eval()

    hnet = UNet(1, 3)
    hnet.load_state_dict(torch.load(os.path.join(opt.m, "HNet/HNet.pth")))
    hnet.to(device)
    hnet.eval()
    return cnet, ssnet, pnet_c, pnet_s, pnet_a, hnet