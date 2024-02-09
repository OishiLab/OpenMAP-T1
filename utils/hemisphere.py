import torch
from scipy.ndimage import binary_dilation
from utils.functions import normalize


def separate(voxel, model, device):
    model.eval()
    with torch.inference_mode():
        output = torch.zeros(256, 3, 256, 256).to(device)
        for i, v in enumerate(voxel):
            image = torch.tensor(v.reshape(1, 1, 256,256))
            image = image.to(device)
            x_out = torch.softmax(model(image),1).detach()
            if i == 0:
                output[0] = x_out
            else:
                output[i] = x_out
        return output
    
def hemisphere(voxel, hnet, device):
    voxel = normalize(voxel)
    
    coronal = voxel.transpose(1, 2, 0)
    axial = voxel.transpose(2, 1, 0)
    out_c = separate(coronal, hnet, device).permute(1,3,0,2)
    out_a = separate(axial, hnet, device).permute(1,3,2,0)
    out_e = out_c + out_a
    out_e = torch.argmax(out_e, 0).cpu().numpy()
    torch.cuda.empty_cache()
    
    dilated_mask_1 = binary_dilation(out_e == 1, iterations=5).astype("int16")
    dilated_mask_1[out_e == 2] = 2
    dilated_mask_2 = binary_dilation(dilated_mask_1 == 2, iterations=5).astype("int16")*2
    dilated_mask_2[dilated_mask_1 == 1] = 1
    return dilated_mask_2