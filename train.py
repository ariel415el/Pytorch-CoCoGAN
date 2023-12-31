from time import time

import torch

from losses import get_loss_function
from utils.diffaug import DiffAugment
from utils.common import dump_images, compose_experiment_name
from utils.patch_generation import generate, generate_full
from utils.train_utils import copy_G_params, load_params, get_models_and_optimizers, parse_train_args, save_model
from utils.data import get_dataloader
from utils.logger import get_dir, PLTLogger, WandbLogger


def train_GAN(args):
    prior, netG, netD, optimizerG, optimizerD, start_iteration = get_models_and_optimizers(args, device, saved_model_folder)

    debug_fixed_noise = prior.sample(args.batch_size)
    debug_fixed_reals, debug_fixed_macro_coords = next(iter(train_loader))

    loss_function = get_loss_function(args.loss_function)

    avg_param_G = copy_G_params(netG)

    start = time()
    iteration = start_iteration
    while iteration < args.n_iterations:
        for macro_patch, top_left_coord in train_loader:
            macro_patch = macro_patch.to(device)

            noise = prior.sample(args.batch_size).to(device)
            fake_images = generate(netG, noise, top_left_coord, args)

            macro_patch = DiffAugment(macro_patch, policy=args.augmentation)
            fake_images = DiffAugment(fake_images, policy=args.augmentation)

            # #####  1. train Discriminator #####
            Dloss, debug_Dlosses = loss_function.trainD(netD, macro_patch, fake_images)
            netD.zero_grad()
            Dloss.backward()
            optimizerD.step()

            logger.log(debug_Dlosses, step=iteration)

            # #####  2. train Generator #####
            if iteration % args.G_step_every == 0:
                if not args.no_fake_resample:
                    noise = prior.sample(args.batch_size).to(device)
                    fake_images = generate(netG, noise, top_left_coord, args)
                    fake_images = DiffAugment(fake_images, policy=args.augmentation)

                Gloss, debug_Glosses = loss_function.trainG(netD, macro_patch, fake_images)
                netG.zero_grad()
                Gloss.backward()
                optimizerG.step()
                logger.log(debug_Glosses, step=iteration)

            # Update avg weights
            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(1 - args.avg_update_factor).add_(args.avg_update_factor * p.data)

            if iteration % 100 == 0:
                it_sec = max(1, iteration - start_iteration) / (time() - start)
                print(f"Iteration: {iteration}: it/sec: {it_sec:.1f}")
                logger.plot()

            if iteration % args.log_freq == 0:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)

                evaluate(netG, debug_fixed_noise, debug_fixed_reals, saved_image_folder, iteration)

                save_model(prior, netG, netD, optimizerG, optimizerD, saved_model_folder, iteration, args)

                load_params(netG, backup_para)

            iteration += 1


def evaluate(netG, fixed_noise, debug_fixed_reals, saved_image_folder, iteration):
    netG.eval()
    start = time()
    with torch.no_grad():
        dump_images(generate_full(netG, fixed_noise.to(device), args),  f'{saved_image_folder}/{iteration}.png')
        dump_images(generate(netG, fixed_noise.to(device), None, args),  f'{saved_image_folder}/{iteration}_macro_path.png')

        if iteration == 0:
            dump_images(debug_fixed_reals, f'{saved_image_folder}/debug_fixed_reals.png')

    netG.train()
    print(f"Evaluation finished in {time()-start} seconds")


if __name__ == "__main__":
    args = parse_train_args()

    if args.train_name is None:
        args.train_name = compose_experiment_name(args)

    saved_model_folder, saved_image_folder, plots_image_folder = get_dir(args)

    logger = (WandbLogger if args.wandb else PLTLogger)(args, plots_image_folder)

    device = torch.device(args.device)
    if args.device != 'cpu':
        print(f"Working on device: {torch.cuda.get_device_name(device)}")

    train_loader = get_dataloader(args.data_path, args.im_size, args.batch_size, args.n_workers, args)

    train_GAN(args)



