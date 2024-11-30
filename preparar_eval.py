import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
import torch 
from torch import nn
from models import FSRCNN
import torch.backends.cudnn as cudnn

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

def eval(args):
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

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        print(f"Procesando {image_path}")
        hr = pil_image.open(image_path).convert('RGB')
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


        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--attack', action='store_true', help='Habilita el ataque iFGSM')
    parser.add_argument('--alpha', type=float, default=0.03137255)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--weights-file', type=str)

    args = parser.parse_args()

    eval(args)