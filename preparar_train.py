import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import calc_patch_size, convert_rgb_to_y
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from models import FSRCNN

def i_fgsm_ataque(modelo, X0, alpha, T, device):
    X = X0.clone().detach().requires_grad_(True).to(device)
    
    with torch.no_grad():
        f_X0 = modelo(X0)

    for _ in range(T):
        loss = nn.MSELoss()(modelo(X), f_X0)
        modelo.zero_grad()
        loss.backward()
        sign_gradient = torch.sign(X.grad)
        X = (X + alpha * sign_gradient).clamp(0, 1)
        X = (X - X0).clamp(-alpha, alpha) + X0
        X = X.detach().requires_grad_(True).to(device)

    return X


@calc_patch_size
def train(args):
    if args.attack:
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("CUDA")
        else:
            print("CPU")
        
        model = FSRCNN(scale_factor=args.scale).to(device)
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict:
                state_dict[n].copy_(p)

        model.eval()

    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        print(f"Procesando {image_path}")
        hr = pil_image.open(image_path).convert('RGB')
        hr_images = []

        if args.with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)
            

            if args.attack:
                lr /= 255.
                lr = torch.from_numpy(lr).to(device)
                lr = lr.unsqueeze(0).unsqueeze(0)
                lr = i_fgsm_ataque(model, lr, args.alpha, args.T, device)
                lr = lr.squeeze().detach().cpu().numpy()
                lr = np.clip(lr * 255, 0, 255).astype(np.float32)

            for i in range(0, lr.shape[0] - args.patch_size + 1, args.scale):
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.scale):
                    lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
                    hr_patches.append(hr[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--with-aug', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.03137255)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--attack', action='store_true', help='Habilita el ataque iFGSM')
    args = parser.parse_args()

    train(args)