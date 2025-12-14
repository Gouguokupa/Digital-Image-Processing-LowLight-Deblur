import os, glob, argparse, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(c, eps=1e-6, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(c, eps=1e-6, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)

class FPNHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FPNGenerator(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.pyr_c = nf * 2

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, nf, 7, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(nf, eps=1e-6, affine=True),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf*2, eps=1e-6, affine=True),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf*4, eps=1e-6, affine=True),
            nn.ReLU(inplace=True)
        )

        self.res = nn.Sequential(*[ResidualBlock(nf*4) for _ in range(9)])

        self.lat3 = nn.Conv2d(nf*4, self.pyr_c, 1)
        self.lat2 = nn.Conv2d(nf*2, self.pyr_c, 1)
        self.lat1 = nn.Conv2d(nf, self.pyr_c, 1)

        self.head1 = FPNHead(self.pyr_c)
        self.head2 = FPNHead(self.pyr_c)
        self.head3 = FPNHead(self.pyr_c)

        self.out1 = nn.Sequential(nn.Conv2d(self.pyr_c, 3, 7, padding=3, padding_mode="reflect"), nn.Tanh())
        self.out2 = nn.Sequential(nn.Conv2d(self.pyr_c, 3, 7, padding=3, padding_mode="reflect"), nn.Tanh())
        self.out3 = nn.Sequential(nn.Conv2d(self.pyr_c, 3, 7, padding=3, padding_mode="reflect"), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        r = self.res(e3)

        p3 = self.lat3(r)
        p2 = self.lat2(e2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p1 = self.lat1(e1) + F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)

        return {
            "scale1": self.out1(self.head1(p1)),
            "scale2": self.out2(self.head2(p2)),
            "scale3": self.out3(self.head3(p3)),
        }

def SN(layer):
    return nn.utils.spectral_norm(layer)

class PatchD(nn.Module):
    def __init__(self, in_ch=3, nf=64, use_spectral_norm=True):
        super().__init__()
        C1 = nf
        C2 = nf*2
        C3 = nf*4
        C4 = nf*8

        def conv(in_c, out_c, k=4, s=2, p=1):
            layer = nn.Conv2d(in_c, out_c, k, s, p)
            return SN(layer) if use_spectral_norm else layer

        self.net = nn.Sequential(
            conv(in_ch, C1, 4, 2, 1), nn.LeakyReLU(0.2, True),
            conv(C1, C2, 4, 2, 1), nn.InstanceNorm2d(C2, eps=1e-6, affine=True), nn.LeakyReLU(0.2, True),
            conv(C2, C3, 4, 2, 1), nn.InstanceNorm2d(C3, eps=1e-6, affine=True), nn.LeakyReLU(0.2, True),
            conv(C3, C4, 4, 1, 1), nn.InstanceNorm2d(C4, eps=1e-6, affine=True), nn.LeakyReLU(0.2, True),
            conv(C4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

class DoubleScaleD(nn.Module):
    def __init__(self, use_spectral_norm=True):
        super().__init__()
        self.full = PatchD(use_spectral_norm=use_spectral_norm)
        self.half_scale = PatchD(use_spectral_norm=use_spectral_norm)

    def forward(self, x):
        out_full = self.full(x)
        x_half = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        out_half = self.half_scale(x_half)
        return out_full, out_half

class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights="DEFAULT").features.eval()
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27]
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2
        loss = 0.0
        for b in self.blocks:
            x = b(x)
            y = b(y)
            loss = loss + F.mse_loss(x, y)
        return loss

def gradient_penalty(d_full, d_half_scale, real, fake):
    B = real.size(0)
    a = torch.rand(B, 1, 1, 1, device=real.device)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    xh = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

    out = torch.cat([
        d_full(x).view(B, -1),
        d_half_scale(xh).view(B, -1)
    ], dim=1)

    g = torch.autograd.grad(
        outputs=out,
        inputs=x,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gp = ((g.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
    return gp

class DeblurDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, image_size=256, augment=False):
        blur_files = sorted(glob.glob(os.path.join(blur_dir, "*.png")) + glob.glob(os.path.join(blur_dir, "*.jpg")))
        sharp_files = sorted(glob.glob(os.path.join(sharp_dir, "*.png")) + glob.glob(os.path.join(sharp_dir, "*.jpg")))

        sharp_map = {os.path.basename(f): f for f in sharp_files}
        self.pairs = []
        for b in blur_files:
            bn = os.path.basename(b)
            if bn in sharp_map:
                self.pairs.append((b, sharp_map[bn]))

        self.image_size = image_size
        self.augment = augment
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        
        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")
        
        min_h = min(blur_img.height, sharp_img.height)
        min_w = min(blur_img.width, sharp_img.width)
        blur_img = blur_img.crop((0, 0, min_w, min_h))
        sharp_img = sharp_img.crop((0, 0, min_w, min_h))
        
        if self.augment:
            do_hflip = torch.rand(1).item() < 0.5
            do_vflip = torch.rand(1).item() < 0.3
        else:
            do_hflip = False
            do_vflip = False
        
        if do_hflip:
            blur_img = blur_img.transpose(Image.FLIP_LEFT_RIGHT)
            sharp_img = sharp_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if do_vflip:
            blur_img = blur_img.transpose(Image.FLIP_TOP_BOTTOM)
            sharp_img = sharp_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        blur_img = blur_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        sharp_img = sharp_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        blur_tensor = self.normalize(self.to_tensor(blur_img))
        sharp_tensor = self.normalize(self.to_tensor(sharp_img))
        
        return blur_tensor, sharp_tensor

class Trainer:
    def __init__(self, device, use_spectral_norm=True, use_amp=False,
                 lambda_adv=0.01, lambda_l1=1.0, lambda_perc=0.1, lambda_ms=0.5,
                 lr=1e-4, gp_weight=1.0):
        self.device = device
        self.G = FPNGenerator().to(device)
        self.D = DoubleScaleD(use_spectral_norm=use_spectral_norm).to(device)

        self.optG = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))

        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptual().to(device)

        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ms = lambda_ms
        self.gp_weight = gp_weight

        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            try:
                self.scalerG = torch.amp.GradScaler('cuda')
                self.scalerD = torch.amp.GradScaler('cuda')
            except:
                self.scalerG = torch.cuda.amp.GradScaler()
                self.scalerD = torch.cuda.amp.GradScaler()

    def train_step(self, blur, sharp):
        blur, sharp = blur.to(self.device), sharp.to(self.device)

        self.optD.zero_grad(set_to_none=True)

        if self.use_amp:
            try:
                with torch.amp.autocast('cuda'):
                    fake = self.G(blur)["scale1"].detach()
                    r_f, r_h = self.D(sharp)
                    f_f, f_h = self.D(fake)

                    d_loss = -(r_f.mean() + r_h.mean()) + (f_f.mean() + f_h.mean())
                    gp = gradient_penalty(self.D.full, self.D.half_scale, sharp, fake)
                    d_total = d_loss + self.gp_weight * gp
            except:
                with torch.cuda.amp.autocast():
                    fake = self.G(blur)["scale1"].detach()
                    r_f, r_h = self.D(sharp)
                    f_f, f_h = self.D(fake)

                    d_loss = -(r_f.mean() + r_h.mean()) + (f_f.mean() + f_h.mean())
                    gp = gradient_penalty(self.D.full, self.D.half_scale, sharp, fake)
                    d_total = d_loss + self.gp_weight * gp

            self.scalerD.scale(d_total).backward()
            self.scalerD.unscale_(self.optD)
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
            self.scalerD.step(self.optD)
            self.scalerD.update()
        else:
            fake = self.G(blur)["scale1"].detach()
            r_f, r_h = self.D(sharp)
            f_f, f_h = self.D(fake)

            d_loss = -(r_f.mean() + r_h.mean()) + (f_f.mean() + f_h.mean())
            gp = gradient_penalty(self.D.full, self.D.half_scale, sharp, fake)
            d_total = d_loss + self.gp_weight * gp

            d_total.backward()
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
            self.optD.step()

        self.optG.zero_grad(set_to_none=True)

        if self.use_amp:
            try:
                with torch.amp.autocast('cuda'):
                    outs = self.G(blur)
                    fake1, fake2, fake3 = outs["scale1"], outs["scale2"], outs["scale3"]

                    f_f, f_h = self.D(fake1)
                    adv = -(f_f.mean() + f_h.mean()) / 2.0

                    l1 = self.l1(fake1, sharp)
                    perc = self.perc(fake1, sharp)

                    sharp2 = F.interpolate(sharp, scale_factor=0.5, mode="bilinear", align_corners=False)
                    sharp3 = F.interpolate(sharp, scale_factor=0.25, mode="bilinear", align_corners=False)
                    ms = self.l1(fake2, sharp2) + self.l1(fake3, sharp3)

                    g_total = (self.lambda_adv * adv +
                               self.lambda_l1 * l1 +
                               self.lambda_perc * perc +
                               self.lambda_ms * ms)
            except:
                with torch.cuda.amp.autocast():
                    outs = self.G(blur)
                    fake1, fake2, fake3 = outs["scale1"], outs["scale2"], outs["scale3"]

                    f_f, f_h = self.D(fake1)
                    adv = -(f_f.mean() + f_h.mean()) / 2.0

                    l1 = self.l1(fake1, sharp)
                    perc = self.perc(fake1, sharp)

                    sharp2 = F.interpolate(sharp, scale_factor=0.5, mode="bilinear", align_corners=False)
                    sharp3 = F.interpolate(sharp, scale_factor=0.25, mode="bilinear", align_corners=False)
                    ms = self.l1(fake2, sharp2) + self.l1(fake3, sharp3)

                    g_total = (self.lambda_adv * adv +
                               self.lambda_l1 * l1 +
                               self.lambda_perc * perc +
                               self.lambda_ms * ms)

            self.scalerG.scale(g_total).backward()
            self.scalerG.unscale_(self.optG)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.scalerG.step(self.optG)
            self.scalerG.update()
        else:
            outs = self.G(blur)
            fake1, fake2, fake3 = outs["scale1"], outs["scale2"], outs["scale3"]

            f_f, f_h = self.D(fake1)
            adv = -(f_f.mean() + f_h.mean()) / 2.0

            l1 = self.l1(fake1, sharp)
            perc = self.perc(fake1, sharp)

            sharp2 = F.interpolate(sharp, scale_factor=0.5, mode="bilinear", align_corners=False)
            sharp3 = F.interpolate(sharp, scale_factor=0.25, mode="bilinear", align_corners=False)
            ms = self.l1(fake2, sharp2) + self.l1(fake3, sharp3)

            g_total = (self.lambda_adv * adv +
                       self.lambda_l1 * l1 +
                       self.lambda_perc * perc +
                       self.lambda_ms * ms)

            g_total.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.optG.step()

        return {
            "d_total": float(d_total.detach().cpu()),
            "d_loss": float(d_loss.detach().cpu()),
            "gp": float(gp.detach().cpu()),
            "g_total": float(g_total.detach().cpu()),
            "adv": float(adv.detach().cpu()),
            "l1": float(l1.detach().cpu()),
            "perc": float(perc.detach().cpu()),
            "ms": float(ms.detach().cpu()),
        }

    @torch.no_grad()
    def evaluate(self, loader):
        self.G.eval()
        total_psnr, total_ssim, n = 0.0, 0.0, 0

        for blur, sharp in loader:
            blur, sharp = blur.to(self.device), sharp.to(self.device)
            pred = self.G(blur)["scale1"]

            for i in range(pred.size(0)):
                p = pred[i].detach().cpu().numpy().transpose(1,2,0)
                t = sharp[i].detach().cpu().numpy().transpose(1,2,0)

                p = ((p + 1) * 127.5).clip(0,255).astype(np.uint8)
                t = ((t + 1) * 127.5).clip(0,255).astype(np.uint8)

                total_psnr += psnr(t, p, data_range=255)
                total_ssim += ssim(t, p, channel_axis=2, data_range=255)
                n += 1

        self.G.train()
        return {"psnr": total_psnr / max(n,1), "ssim": total_ssim / max(n,1)}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = DeblurDataset(args.train_blur, args.train_sharp, args.image_size, augment=True)
    val_ds   = DeblurDataset(args.val_blur, args.val_sharp, args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    trainer = Trainer(
        device=device,
        use_spectral_norm=True,
        use_amp=args.use_amp,
        lambda_adv=args.lambda_adv,
        lambda_l1=args.lambda_l1,
        lambda_perc=args.lambda_perc,
        lambda_ms=args.lambda_ms,
        lr=args.lr,
        gp_weight=args.gp_weight
    )

    best_psnr = -1.0

    for epoch in range(1, args.epochs+1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        running = {k: 0.0 for k in ["d_total","d_loss","gp","g_total","adv","l1","perc","ms"]}
        for blur, sharp in pbar:
            stats = trainer.train_step(blur, sharp)
            for k in running:
                running[k] += stats[k]

            pbar.set_postfix({
                "D": f"{stats['d_total']:.3f}",
                "G": f"{stats['g_total']:.3f}",
                "gp": f"{stats['gp']:.3f}",
                "l1": f"{stats['l1']:.3f}"
            })

        for k in running:
            running[k] /= max(len(train_loader), 1)

        if epoch % args.eval_every == 0:
            metrics = trainer.evaluate(val_loader)
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                path = os.path.join(args.save_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "G": trainer.G.state_dict(),
                    "D": trainer.D.state_dict(),
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "args": vars(args),
                }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_blur", required=True)
    p.add_argument("--train_sharp", required=True)
    p.add_argument("--val_blur", required=True)
    p.add_argument("--val_sharp", required=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="./checkpoints_complete_stable")

    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--eval_every", type=int, default=5)

    p.add_argument("--lambda_adv", type=float, default=0.01)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_perc", type=float, default=0.1)
    p.add_argument("--lambda_ms", type=float, default=0.5)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gp_weight", type=float, default=1.0)

    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
