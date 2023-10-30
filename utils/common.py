import os
from math import sqrt
from torchvision.utils import save_image


def compose_experiment_name(args):
    return f"{os.path.basename(args.data_path)}_I-{args.im_size}x{args.im_size}_G-{args.gen_arch}_D-{args.disc_arch}" \
                f"{'_GS' if args.gray_scale else ''}{f'_CC-{args.center_crop}' if args.center_crop else ''}" \
                f"_L-{args.loss_function}_Z-{args.z_dim}_B-{args.batch_size}"


def parse_classnames_and_kwargs(string, kwargs=None):
    """Import a class and create an instance with kwargs from strings in format
    '<class_name>_<kwarg_1_name>=<kwarg_1_value>_<kwarg_2_name>=<kwarg_2_value>"""
    name_and_args = string.split('-')
    class_name = name_and_args[0]
    if kwargs is None:
        kwargs = dict()
    for arg in name_and_args[1:]:
        name, value = arg.split("=")
        kwargs[name] = value
    return class_name, kwargs


def dump_images(batch, fname):
    nrow = int(sqrt(len(batch)))
    # save_image(batch.add(1).mul(0.5), fname, nrow=nrow, normalize=False)
    save_image(batch, fname, nrow=nrow, normalize=True, pad_value=1, scale_each=True)
