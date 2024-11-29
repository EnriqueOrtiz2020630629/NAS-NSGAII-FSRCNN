import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import FSRCNN
from utils import TrainDataset, EvalDataset, AverageMeter, calc_psnr, calc_patch_size
        
def train_model(d,s,m,outputs_dir,train_file, eval_file,scale=4,lr=1e-3,num_workers=8, batch_size=16,num_epochs=20, seed=123,):
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    model = FSRCNN(scale_factor=scale, d=d,s=s,m=m).to(device)
    weight_name = f"{d}_{s}_{m}.pth"
    weight_path = os.path.join(outputs_dir, weight_name)


    if os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Se cargo pesos de {weight_path}")
        return model
 
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)


    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols=80) as t:
            t.set_description('epoca: {}/{}'.format(epoch, num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
            
            
            model.eval()
            epoch_psnr = AverageMeter()

            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

        print('best epoca: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, os.path.join(outputs_dir, f'{d}_{s}_{m}.pth'))

    model.load_state_dict(torch.load(weight_path, map_location=device))

    return model


def evaluate_model(model, eval_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    epoch_psnr = AverageMeter()

    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model.eval()

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    return  epoch_psnr.avg.item()


if __name__ == "__main__":
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = train_model(d=56, s=12, m=4, num_epochs=2, train_file="Set5train.h5", eval_file="Set5_limpio.h5", outputs_dir="./models")

    print(evaluate_model(model, "Set5_atacado.h5"))




