
import torch
import argparse, logging, copy
from types import SimpleNamespace
from torch import optim
import torch.nn as nn
from fastprogress import progress_bar
import os, random
# import wandb
import numpy as np
from modules.diff import UNet_conditional, EMA
from tasks.dataset_loader.dataset_utils.csv_dataset import CSVDataset_diff
from torch.utils.data import Dataset, DataLoader
from modules.vae import VAEModel

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2 ** 32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = SimpleNamespace(
    run_name="DDPM_conditional",
    data_dir='/root/VAE_rnn/AMP_VAEflow/datasets/combined_dataset_for_diff.csv',
    epochs=50,
    noise_steps=999,
    seed=222,
    batch_size=8,
    seq_len=48,
    num_class1=3,
    num_class2=2,
    channel_size=2,
    train_folder="train",
    val_folder="test",
    device="cuda:2",
    slice_size=1,
    do_validation=True,
    fp16=True,
    log_every_epoch=5,
    num_workers=10,
    lr=5e-3,
    mode='train',
    #mode='infer'
    )

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# [bs, 16, 2, seq_len]
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, seq_len=30, num_class1=2, c_in=2, c_out=2,
                 device=config.device, **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.seq_len = seq_len
        self.model = UNet_conditional(c_in, c_out, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_class1
        self.infer_vae = VAEModel(out_dim=1, con_dim=3, latent_size=16, sqz_prior=False, cond_drop=False, use_prior_flow=True, device=self.device)
        checkpoint = torch.load('VAE_flow.ckpt')
        self.infer_vae.load_state_dict(checkpoint['state_dict']['model'])
        for param in self.infer_vae.parameters():
            param.requires_grad = False

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to sample at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=-1):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new AMP....")
        model.eval()
        with torch.inference_mode():
            maxlen=self.seq_len
            # noise(bs, 2, 16, maxlen)
            x = torch.randn((n, self.c_in, 16, maxlen)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, 
                                                 predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

        return x, maxlen

    def train_step(self, loss):
        self.optimizer.zero_grad()
        loss = loss.to(self.device)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        model_out = {}
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, sample in enumerate(pbar):
            keys = ["idx", "protein", "describe", "point_describe"]
            new_dict = {key: sample[key] for key in keys}
            labels = torch.Tensor(list(map(int, sample['class'])))
            with torch.cuda.amp.autocast() and (torch.inference_mode() if not train else torch.enable_grad()):
                with torch.no_grad():
                    _, m_q, logs_q = self.infer_vae(batch=new_dict, ret=model_out, train=False, temperature=1.)
                combined = torch.stack((m_q, logs_q), dim=2)  # （bs，seq_len，2，z_dim）
                distri = combined.permute(0, 3, 2, 1).to(self.device)  # （bs ,z_dim ,2 ,seq_len）
                labels = labels.to(self.device).long()
                t = self.sample_timesteps(distri.shape[0]).to(self.device)
                x_t, noise = self.noise_images(distri, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t.permute(0, 2, 1, 3), t, labels)
                loss = self.mse(noise.permute(0, 2, 1, 3), predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
            pbar.comment = f"MSE={loss.item():2.3f}"
        return avg_loss.mean().item()


    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch):
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"{epoch}ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"{epoch}ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"{epoch}optim.pt"))


    def prepare(self, args):
        mk_folders(args.run_name)
        csv_dataset = CSVDataset_diff(args.data_dir, num_classes=args.num_class1)
        self.train_dataloader, self.val_dataloader = DataLoader(csv_dataset, batch_size=config.batch_size, shuffle=True), DataLoader(csv_dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _ = self.one_epoch(train=True)

            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
            if epoch!=0 and epoch%10==0:
                self.save_model(run_name=args.run_name, epoch=epoch)




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--data_dir', type=str, default=config.data_dir)
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--seq_len', type=int, default=config.seq_len, help='image size')
    parser.add_argument('--num_class1', type=int, default=config.num_class1, help='number of classes')
    parser.add_argument('--num_class2', type=int, default=config.num_class2, help='number of classes')
    parser.add_argument('--channel_size', type=int, default=config.channel_size)
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--mode', type=str, default=config.mode)
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    task = config.mode

    print(f'Now we perform {task} task')

    ## seed everything
    set_seed(config.seed)
    diffuser = Diffusion(config.noise_steps, seq_len=config.seq_len, num_class1=config.num_class1, 
                         c_in=config.channel_size, c_out=config.channel_size)

    if task == 'train':
        diffuser.prepare(config)
        diffuser.fit(config)

    else:
        checkpoint = torch.load('/rootVAE_rnn/AMP_VAEflow/models/DDPM_conditional/40ema_ckpt.pt')
        diffuser.ema_model.load_state_dict(checkpoint)

        checkpoint = torch.load('/root/VAE_rnn/AMP_VAEflow/models/DDPM_conditional/40ckpt.pt')
        diffuser.model.load_state_dict(checkpoint)

        file_path0 = 'kang_pn_pedi.txt'
        file_path1 = 'kang_n_pedi.txt'
        file_path2 = 'kang_p_pedi.txt'
        sample_nums = 2000

        with open(file_path0, 'a') as file:
            output, lenthout = diffuser.sample(use_ema=True,
                                               labels=torch.zeros(sample_nums).to(config.device).long(),
                                               cfg_scale=-1)
            for i in range(sample_nums):
                x = output[i]  # [2, 16, 30]
                x1, x2 = x.split(1, dim=0)  # [1,16,30]
                m = x1.squeeze(1)  # [16, 30]
                logs = x2.squeeze(1)  # [16, 30]
                z = (m + torch.randn_like(m) * torch.exp(logs))
                output1 = diffuser.infer_vae.vae.infer(z)
                print(output1, len(output1))
                # print(m,logs)
                if len(output1) >= 5:
                    file.write(output1 + '\n')

        with open(file_path1, 'a') as file:
            output, lenthout = diffuser.sample(use_ema=True,
                                               labels=torch.ones(sample_nums).to(config.device).long(),
                                               cfg_scale=-1)
            for i in range(sample_nums):
                x = output[i]  #[2, 16, 30]
                x1, x2 = x.split(1, dim=0)  # [1,16,30]
                m = x1.squeeze(1)  #  [16, 30]
                logs = x2.squeeze(1)  #  [16, 30]
                z = (m + torch.randn_like(m) * torch.exp(logs))
                output1 = diffuser.infer_vae.vae.infer(z)
                print(output1, len(output1))
                # print(m,logs)
                if len(output1) >= 5:
                    file.write(output1 + '\n')

        with open(file_path2, 'a') as file:
            output, lenthout = diffuser.sample(use_ema=True,
                                               labels=torch.full((sample_nums,), 2).to(config.device).long(),
                                               cfg_scale=-1)
            for i in range(sample_nums):
                x = output[i]
                x1, x2 = x.split(1, dim=0)
                m = x1.squeeze(1)
                logs = x2.squeeze(1)
                z = (m + torch.randn_like(m) * torch.exp(logs))
                output1 = diffuser.infer_vae.vae.infer(z)
                print(output1, len(output1))
                # print(m,logs)
                if len(output1) >= 5:
                    file.write(output1 + '\n')
