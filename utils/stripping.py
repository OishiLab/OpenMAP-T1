import torch
from utils.functions import normalize


def strip(voxel, model, device):
    model.eval()
    with torch.inference_mode():
        output = torch.zeros(256, 256, 256).to(device)
        for i, v in enumerate(voxel):
            image = v.reshape(1,1,256,256)
            image = torch.tensor(image).to(device)
            x_out = torch.sigmoid(model(image)).detach()
            if i == 0:
                output[0] = x_out
            else:
                output[i] = x_out
        return output.reshape(256, 256, 256)
    
def stripping(voxel, data, ssnet, device):
    voxel = normalize(voxel)
    
    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)
    out_c = strip(coronal, ssnet, device).permute(2,0,1)
    out_s = strip(sagittal, ssnet, device)
    out_a = strip(axial, ssnet, device).permute(2,1,0)
    out_e = ((out_c + out_s + out_a) / 3) > 0.5
    out_e = out_e.cpu().numpy()
    stripped = data.get_fdata() * out_e
    return stripped