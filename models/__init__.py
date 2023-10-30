import importlib

from utils.common import parse_classnames_and_kwargs


def get_models(args, device):
    c = 1 if args.gray_scale else 3
    coord_dim = 2*args.micro_patch_size**2 if args.full_coords else 2
    model_name, kwargs = parse_classnames_and_kwargs(args.gen_arch,
                                                     kwargs={"output_dim": args.micro_patch_size, "z_dim": args.z_dim + coord_dim, "channels": c, "input_dim": args.im_size})
    netG = importlib.import_module("models." + model_name).Generator(**kwargs)

    model_name, kwargs = parse_classnames_and_kwargs(args.disc_arch,
                                                     kwargs={"input_dim": args.macro_patch_size, "channels": c})
    netD = importlib.import_module("models." + model_name).Discriminator(**kwargs)

    return netG.to(device), netD.to(device)

