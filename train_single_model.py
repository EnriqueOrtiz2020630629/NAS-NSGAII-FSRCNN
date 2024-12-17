import argparse
from train import train_model, evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y evalua una arquitectura en especifico")
    
    parser.add_argument('--d', type=int, default=56)
    parser.add_argument('--s', type=int, default=12)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_file', type=str, default="T91atacado.h5")
    parser.add_argument('--eval_file', type=str, default="Set5_limpio.h5")
    parser.add_argument('--outputs_dir', type=str, default="./models")
    args = parser.parse_args()
    

    model = train_model(d=56,s=12,m=4, num_workers=8, num_epochs=args.num_epochs, train_file="T91atacado.h5", eval_file="Set5_limpio.h5", outputs_dir="./models")

    psnr_limpio = evaluate_model(model, "Set5_limpio.h5")
    psnr_atacado = evaluate_model(model, "Set5_atacado.h5")

    print(f"Modelo: [{args.d}, {args.s}, {args.m}]")
    print(f"PSNR limpio: {psnr_limpio}")
    print(f"PSNR atacado: {psnr_atacado}")


