#!/usr/bin/env python3

"""Generates images from text prompts with VQGAN and CLIP."""

import argparse
from concurrent import futures
import sys

# pip install taming-transformers works with Gumbel, but does not yet work with coco etc
# appending the path works with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
from taming.modules.diffusionmodules.model import Decoder
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from CLIP import clip
from resample import resample


def setup_exceptions():
    try:
        from IPython.core.ultratb import FormattedTB
        sys.excepthook = FormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def spherical_dist(x, y):
    x_normed = F.normalize(x, dim=-1)
    y_normed = F.normalize(y, dim=-1)
    return x_normed.sub(y_normed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        dists = spherical_dist(input.unsqueeze(1), self.embed.unsqueeze(0))
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


class MPDecoder(Decoder):
    def __init__(self, devices, **kwargs):
        assert len(devices) == 2
        self.devices = devices
        super().__init__(**kwargs)
        self.to(devices[0])
        for module in self.up[0].block[1:]:
            module.to(devices[1])
        self.norm_out.to(devices[1])
        self.conv_out.to(devices[1])

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                if i_level == 0 and i_block == 1:
                    h = h.to(self.devices[1])
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = h * torch.sigmoid(h)
        h = self.conv_out(h)
        return h


def load_vqgan_model(config_path, checkpoint_path, devices):
    assert len(devices) == 2
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        ddconfig = config.model.params.ddconfig
        model = vqgan.VQModel(**config.model.params)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        ddconfig = config.model.params.ddconfig
        model = vqgan.GumbelVQ(**config.model.params)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        ddconfig = config.model.params.first_stage_config.params.ddconfig
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    # del model.loss
    model.to(devices[0])
    mp_decoder = MPDecoder(devices, **ddconfig)
    mp_decoder.load_state_dict(model.decoder.state_dict())
    model.decoder = mp_decoder
    model.requires_grad_(False).eval()
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def save_image(t, name):
    TF.to_pil_image(t).save(name)


def main():
    setup_exceptions()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompts', type=str, default=[], nargs='*',
                   help='the text prompts')
    p.add_argument('--images', type=str, default=[], nargs='*', metavar='IMAGE',
                   help='the image prompts')
    p.add_argument('--noise-prompts', '-np', type=str, default=[], nargs='*', metavar='PROMPT',
                   help='the noise prompts\' random seeds, weights, and stop losses')
    p.add_argument('--spatial-images', type=str, default=[], nargs='*', metavar='IMAGE',
                   help='the spatial image prompts')
    p.add_argument('--devices', type=str, default=[], nargs='+',
                   help='the device names to use (omit for auto)')
    p.add_argument('--size', '-s', type=int, default=[512, 512], nargs=2,
                   help='the image size')
    p.add_argument('--init', type=str,
                   help='the init image')
    p.add_argument('--init-weight', '-iw', type=float, default=0.,
                   help='the weight for the distance from the init')
    p.add_argument('--mask', type=str,
                   help='the mask to use')
    p.add_argument('--invert-mask', action='store_true',
                   help='invert the mask')
    p.add_argument('--soft-mask', action='store_true',
                   help='use masked init weight instead of a hard mask')
    p.add_argument('--clip-model', type=str, default='ViT-B/32', choices=clip.available_models(),
                   help='the CLIP model to use')
    p.add_argument('--vqgan-model', type=str, default='checkpoints/vqgan_imagenet_f16_16384',
                   help='the VQGAN model')
    p.add_argument('--step-size', '-ss', type=float, default=0.05,
                   help='the step size')
    p.add_argument('--cutn', type=int, default=64,
                   help='the number of cutouts')
    p.add_argument('--cut-pow', type=float, default=1.,
                   help='the cutout size power')
    p.add_argument('--display-freq', type=int, default=25,
                   help='display every this many steps')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    args = p.parse_args()

    devices = [torch.device(device) for device in args.devices]
    if not devices:
        devices = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')]
    if not 1 <= len(devices) <= 2:
        print('Only 1 or 2 devices are supported.')
        sys.exit(1)
    print('Using devices:', ', '.join(str(device) for device in devices))
    if len(devices) == 1:
        devices = devices * 2

    model = load_vqgan_model(args.vqgan_model + '.yaml', args.vqgan_model + '.ckpt', devices)
    perceptor = clip.load(args.clip_model, jit=False)[0]
    perceptor.to(devices[1]).eval().requires_grad_(False)
    pool = futures.ThreadPoolExecutor()

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    torch.manual_seed(args.seed)

    if args.init:
        pil_image = Image.open(args.init).convert('RGB')
        pil_image = pil_image.resize((toksX * f, toksY * f), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(devices[0]).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=devices[0]), n_toks)
        z = one_hot.float() @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_()
    opt = optim.Adam([z], lr=args.step_size)

    if args.mask:
        pil_image = Image.open(args.mask)
        if 'A' in pil_image.getbands():
            pil_image = pil_image.getchannel('A')
        elif 'L' in pil_image.getbands():
            pil_image = pil_image.getchannel('L')
        else:
            print('Mask must have an alpha channel or be one channel', file=sys.stderr)
            sys.exit(1)
        mask = TF.to_tensor(pil_image.resize((toksX, toksY), Image.BILINEAR))
        mask = mask.to(devices[0]).unsqueeze(0)
        if not args.soft_mask:
            mask = mask.lt(0.5).float()
    else:
        mask = torch.ones([], device=devices[0])
    if args.invert_mask:
        mask = 1 - mask

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(devices[1])).float()
        pMs.append(Prompt(embed, weight, stop).to(devices[1]))

    for prompt in args.images:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(devices[1]))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(devices[1]))

    for prompt in args.noise_prompts:
        seed, weight, stop = parse_prompt(prompt)
        gen = torch.Generator().manual_seed(int(seed))
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight, stop).to(devices[1]))

    spatial_images = []

    for prompt in args.spatial_images:
        path, weight, stop = parse_prompt(prompt)
        pil_image = Image.open(path).convert('RGB').resize((sideX, sideY), Image.LANCZOS)
        image = TF.to_tensor(pil_image).unsqueeze(0).to(devices[1])
        spatial_images.append((image, weight, stop))

    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(i, losses):
        losses_i = [loss.item() for loss in losses]
        losses_str = ', '.join(f'{loss:g}' for loss in losses_i)
        tqdm.write(f'i: {i}, loss: {sum(losses_i):g}, losses: {losses_str}')
        out = synth(z)
        pool.submit(save_image, out[0], f'out_{i:05}.png')

    def ascend_txt():
        out = synth(z if args.soft_mask else replace_grad(z, z * mask))
        seed = torch.randint(2**63 - 1, [])

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        si_embeds = []
        for image, weight, stop in spatial_images:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                si_embed = perceptor.encode_image(normalize(make_cutouts(image))).float()
            si_embeds.append((si_embed, torch.tensor(weight), torch.tensor(stop)))

        result = []

        if args.init_weight:
            diffs = F.mse_loss(z, z_orig, reduction='none')
            if args.soft_mask:
                diffs = diffs * mask
            result.append(diffs.mean() * args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        for embeds, weight, stop in si_embeds:
            dists = spherical_dist(iii, embeds) * weight.sign()
            result.append(weight.abs() * replace_grad(dists, torch.maximum(dists, stop)).mean())

        return result

    def train(i):
        opt.zero_grad()
        loss_all = ascend_txt()
        loss_all_d = [loss.to(loss_all[0].device) for loss in loss_all]
        if i % args.display_freq == 0:
            checkin(i, loss_all_d)
        loss = sum(loss_all_d)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    i = 0
    try:
        with tqdm() as pbar:
            while True:
                train(i)
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
