import argparse
import glob
import os
from copy import deepcopy
import torch
from torch import optim as optim

from models import get_models


def parse_train_args(arguments_string=None):
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ_1000/FFHQ_1000",
                        help="Path to train images")
    parser.add_argument('--augmentation', default='color,translation,horizontal_flip', help="comma separated data augmentation ('color,translation,horizontal_flip')")
    parser.add_argument('--limit_data', default=None, type=int, help="limit the size of the dataset")
    parser.add_argument('--center_crop', default=None, help='center_crop_data to specified size', type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False, help="Load data as grayscale")

    # Model
    parser.add_argument('--gen_arch', default='DCGAN')
    parser.add_argument('--disc_arch', default='DCGAN')
    parser.add_argument('--im_size', default=128, type=int)
    parser.add_argument('--z_dim', default=128, type=int)
    parser.add_argument('--full_coords', action='store_true', default=False)
    parser.add_argument('--macro_patch_size', default=64, type=int)
    parser.add_argument('--micro_patch_size', default=32, type=int)


    # Training
    parser.add_argument('--batch_size', default=64, type=int, help="Real data batch size")
    parser.add_argument('--loss_function', default="SoftHingeLoss", type=str)
    parser.add_argument('--lrG', default=0.0002, type=float)
    parser.add_argument('--lrD', default=0.0002
                        , type=float)
    parser.add_argument('--avg_update_factor', default=1, type=float,
                        help='moving average factor weight of updating generator (1 means none)')
    parser.add_argument('--G_step_every', default=1, type=int, help="Update G only evry 'G_step_every' iterations")
    parser.add_argument('--n_iterations', default=1000000, type=int)
    parser.add_argument('--no_fake_resample', default=False, action='store_true')

    # Evaluation
    parser.add_argument('--wandb', action='store_true', default=False, help="Otherwise use PLT localy")
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--save_every', action='store_true', default=False)

    # Other
    parser.add_argument('--project_name', default='GANs')
    parser.add_argument('--train_name', default=None)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--resume_last_ckpt', action='store_true', default=False,
                        help="Search for the latest ckpt in the same folder to resume training")
    parser.add_argument('--device', default="cuda:0")

    if arguments_string is not None:
        arguments_string = arguments_string.split()

    return parser.parse_args(arguments_string)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


class Prior:
    def __init__(self, prior_type, z_dim):
        self.prior_type = prior_type
        self.z_dim = z_dim
        self.z = None

    def sample(self, b):
        if self.prior_type == "uniform":
            z = torch.rand((b, self.z_dim))
        else:
            z = torch.randn((b, self.z_dim))
        return z


def save_model(prior, netG, netD, optimizerG, optimizerD, saved_model_folder, iteration, args):
    fname = f"{saved_model_folder}/{'last' if not args.save_every else iteration}.pth"
    torch.save({"iteration": iteration,
                'prior': prior.z,
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict()
                },
               fname)


def load_ckpt(ckpt_path, netG, netD, optimizerG, optimizerD, prior, args):
    ckpts = glob.glob(ckpt_path)
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        ckpt = torch.load(latest_ckpt, map_location=args.device)
        prior.z = ckpt['prior']
        netG.load_state_dict(ckpt['netG'])
        netD.load_state_dict(ckpt['netD'])
        optimizerG.load_state_dict(ckpt['optimizerG'])
        optimizerD.load_state_dict(ckpt['optimizerD'])
        start_iteration = ckpt['iteration']
        print(f"Loaded ckpt of iteration: {start_iteration}")
    return netG, netD, optimizerG, optimizerD, prior


def get_models_and_optimizers(args, device, saved_model_folder):
    prior = Prior("normal", args.z_dim)

    netG, netD = get_models(args, device)
    netG.train()
    netD.train()

    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.999))

    start_iteration = 0
    if args.resume_last_ckpt:
        netG, netD, optimizerG, optimizerD, prior = load_ckpt(f'{saved_model_folder}/*.pth', netG, netD, optimizerG, optimizerD, prior, args)

    return prior, netG, netD, optimizerG, optimizerD, start_iteration