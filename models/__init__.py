import importlib

from utils.common import parse_classnames_and_kwargs


def get_models(args, device):
    c = 1 if args.gray_scale else 3
    model_name, kwargs = parse_classnames_and_kwargs(args.gen_arch,
                                                     kwargs={"output_dim": args.im_size, "z_dim": args.z_dim, "channels": c, "input_dim": args.im_size})
    netG = importlib.import_module("models." + model_name).Generator(**kwargs)

    model_name, kwargs = parse_classnames_and_kwargs(args.disc_arch,
                                                     kwargs={"input_dim": args.im_size, "channels": c})
    netD = importlib.import_module("models." + model_name).Discriminator(**kwargs)

    return netG.to(device), netD.to(device)

