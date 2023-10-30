import numpy as np
import torch


def cat2x2(tl_patch, tr_patch, bl_patch, br_patch):
    return torch.cat([torch.cat([tl_patch, tr_patch], dim=-1),
                             torch.cat([bl_patch, br_patch], dim=-1)],
                             dim=-2)


def generate_full(netG, noise, args):
    b = noise.shape[0]
    top_left_coord = torch.zeros(b,2)
    tl_macro_patch = generate(netG, noise, top_left_coord, args)
    tr_macro_patch = generate(netG, noise, top_left_coord + torch.tensor([0, args.macro_patch_size]), args)
    bl_macro_patch = generate(netG, noise, top_left_coord + torch.tensor([args.macro_patch_size, 0]), args)
    br_macro_patch = generate(netG, noise, top_left_coord + args.macro_patch_size, args)
    full_image = cat2x2(tl_macro_patch, tr_macro_patch, bl_macro_patch, br_macro_patch)
    return full_image


def generate(netG, noise, top_left_coord, args):
    b = noise.shape[0]
    if top_left_coord is None:
        top_left_coord = torch.tensor([args.macro_patch_size, args.macro_patch_size]).unsqueeze(0).repeat(b, 1)
    device = noise.device
    if args.full_coords:
        coord_range = np.arange(args.micro_patch_size)
        base_micro_coords = torch.from_numpy(np.stack(np.meshgrid(coord_range, coord_range))[::-1].copy())
        base_micro_coords.unsqueeze(0)
        tl_micro_coords = base_micro_coords + top_left_coord.unsqueeze(-1).unsqueeze(-1)
        tr_micro_coords = base_micro_coords + (top_left_coord + torch.tensor([0, args.micro_patch_size])).unsqueeze(-1).unsqueeze(-1)
        bl_micro_coords = base_micro_coords + (top_left_coord + torch.tensor([args.micro_patch_size, 0])).unsqueeze(-1).unsqueeze(-1)
        br_micro_coords = base_micro_coords + (top_left_coord + args.micro_patch_size).unsqueeze(-1).unsqueeze(-1)

        tl_micro_coords = tl_micro_coords.reshape(b, -1).to(device) / (args.macro_patch_size * 2)
        tr_micro_coords = tr_micro_coords.reshape(b, -1).to(device) / (args.macro_patch_size * 2)
        bl_micro_coords = bl_micro_coords.reshape(b, -1).to(device) / (args.macro_patch_size * 2)
        br_micro_coords = br_micro_coords.reshape(b, -1).to(device) / (args.macro_patch_size * 2)

    else:
        tl_micro_coords = top_left_coord.to(device) / (args.macro_patch_size * 2)
        tr_micro_coords = (top_left_coord + torch.tensor([0, args.micro_patch_size])).reshape(b, -1).to(device) / (args.macro_patch_size * 2)
        bl_micro_coords = (top_left_coord + torch.tensor([args.micro_patch_size, 0])).reshape(b, -1).to(device) / (args.macro_patch_size * 2)
        br_micro_coords = (top_left_coord + + torch.tensor([args.micro_patch_size, args.micro_patch_size])).reshape(b, -1).to(device) / (args.macro_patch_size * 2)

    tl_micro_patch = netG(torch.cat([noise, tl_micro_coords], dim=1))
    tr_micro_patch = netG(torch.cat([noise, tr_micro_coords], dim=1))
    bl_micro_patch = netG(torch.cat([noise, bl_micro_coords], dim=1))
    br_micro_patch = netG(torch.cat([noise, br_micro_coords], dim=1))

    generated_macro_patch = cat2x2(tl_micro_patch, tr_micro_patch, bl_micro_patch, br_micro_patch)

    return generated_macro_patch