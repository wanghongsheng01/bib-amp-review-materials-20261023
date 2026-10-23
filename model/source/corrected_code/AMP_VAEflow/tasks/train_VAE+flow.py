
import torch
import torch.nn.functional as F
import numpy as np
import os
from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np
from utils.nn.model_utils import print_arch, get_device_of_model, not_requires_grad
from utils.nn.schedulers import ExponentialSchedule
from utils.nn.grad import get_grad_norm

from modules.vae import VAEModel
# 这地方要把数据集换掉，换成重建的蛋白质序列和其对应的化学描述符序列
from tasks.dataset_loader.dataset_utils.csv_dataset import CSVDataset
from torch.utils.data import DataLoader


PAD_IDX = 20


class train_VAE_flow_Task(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = CSVDataset
        self.bs = 32

    def build_model(self):

        self.model = VAEModel(out_dim=1, con_dim=3, latent_size=16, 
                              sqz_prior=False, cond_drop=False, 
                              use_prior_flow=True, device=hparams['device'])

        return self.model

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        return optimizer

    def build_scheduler(self, optimizer):
        return ExponentialSchedule(optimizer, hparams['lr'], hparams['warmup_updates'])

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(hparams['binary_data_dir'])
        self.train_dl = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        val_dataset = self.dataset_cls(hparams['val_data_dir'])
        self.val_dl = DataLoader(val_dataset, batch_size=self.bs, shuffle=True)
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        val_dataset = self.dataset_cls(hparams['val_data_dir'])
        self.val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return self.val_dl

    ##########################
    # training and validation
    ##########################
    def run_model(self, sample, infer=False, temperature=1.0):
        """
        render or train on a single-frame
        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return:
            if not infer:
                return losses, model_out
            if infer:
                return model_out
        """
        model_out = {}
        if infer:
            self.model(batch=sample, rat=model_out, train=False, temperature=temperature)
        else:
            self.model(sample, model_out, train=True)

        if not infer:
            losses_out = {}
            x_pred = model_out['pred']
            # [mbsize x seq_len x vocabsize]
            amino_acid_to_index = {
                'A': 0,  # Alanine
                'R': 1,  # Arginine
                'N': 2,  # Asparagine
                'D': 3,  # Aspartic acid
                'C': 4,  # Cysteine
                'E': 5,  # Glutamic acid
                'Q': 6,  # Glutamine
                'G': 7,  # Glycine
                'H': 8,  # Histidine
                'I': 9,  # Isoleucine
                'L': 10,  # Leucine
                'K': 11,  # Lysine
                'M': 12,  # Methionine
                'F': 13,  # Phenylalanine
                'P': 14,  # Proline
                'S': 15,  # Serine
                'T': 16,  # Threonine
                'W': 17,  # Tryptophan
                'Y': 18,  # Tyrosine
                'V': 19  # Valine
            }
            indexed_sequences = [[amino_acid_to_index[aa] for aa in seq] for seq in sample['protein']]

            pad_index = PAD_IDX
            max_length = 48
            padded_sequences = [seq + [pad_index] * (max_length - len(seq)) for seq in indexed_sequences]
            x_gt = torch.tensor(padded_sequences).to(hparams['device'])
            x_mask = model_out['mask'].to(hparams['device'])
            losses_out['mse'] = self.mse_loss(x_gt, x_pred, x_mask)
            losses_out['kl'] = model_out['loss_kl']
            return losses_out, model_out
        else:
            return model_out

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'kl': hparams['lambda_kl'],
            'mse': 1.0
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)

    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'],
                                    f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        outputs = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        pred_exp = model_out['pred']
        self.save_result(pred_exp, "pred_exp_val", self.gen_dir)
        if hparams['save_gt']:
            base_fn = f"gt_exp_val"
            self.save_result(sample['exp'], base_fn, self.gen_dir)
        return outputs

    def test_end(self, outputs):
        pass

    @staticmethod
    def save_result(exp_arr, base_fname, gen_dir):
        exp_arr = convert_to_np(exp_arr)
        np.save(f"{gen_dir}/{base_fname}.npy", exp_arr)

    def get_grad(self, opt_idx):
        grad_dict = {
            'grad/model': get_grad_norm(self.model),
        }
        return grad_dict
    
    def recon_dec(self, sequences, logits):
        """ compute reconstruction error (NLL of next-timestep predictions) """
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        # sequences: [mbsize x seq_len]
        # logits: [mbsize x seq_len x vocabsize]
        recon_loss = F.cross_entropy(  # this is log_softmax + nll
            logits.view(-1, logits.size(2)), sequences.view(-1), reduction='mean',
            # ignore_index = PAD_IDX  # padding doesnt contribute to recon loss & gradient
        )
 
        return recon_loss
    
    def mse_loss(self, x_gt, x_pred, x_mask):
        error = (x_pred - x_gt)
        indices_not_equal_to_PAD = torch.nonzero(x_gt != PAD_IDX)
        nums = indices_not_equal_to_PAD.size(0)
        # num_frame =x_gt.shape[0] * x_gt.shape[1]
        return (error ** 2).sum() / nums
    
    def continuity_loss(self, x_gt, x_pred, seq_len):
        # continuity loss, borrowed from <FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning>
        diff_x_pred = x_pred[:, 1:] - x_pred[:, :-1]
        diff_x_gt = x_gt[:, 1:] - x_gt[:, :-1]
        error = diff_x_pred[:, :] - diff_x_gt[:, :]
        init_error = x_pred[:, 0] - x_gt[:, 0]
        num_frame = len(x_gt)
        n_dim = seq_len
        total_error = (error.pow(2).sum() + init_error.pow(2).sum()) / (num_frame * n_dim)
        return total_error


if __name__ == '__main__':
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt

    device='cuda'
    def mse_loss(x_gt, x_pred):
        # mean squared error, l2 loss
        error = (x_pred - x_gt)
        indices_not_equal_to_20 = torch.nonzero(x_gt != 20)
        num_frame = indices_not_equal_to_20.size(0)
        return (error ** 2).sum() / num_frame
    def continuity_loss(x_gt, x_pred, seq_len):
        # continuity loss, borrowed from <FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning>
        diff_x_pred = x_pred[:, 1:] - x_pred[:, :-1]
        diff_x_gt = x_gt[:, 1:] - x_gt[:, :-1]
        error = diff_x_pred[:, :] - diff_x_gt[:, :]
        init_error = x_pred[:, 0] - x_gt[:, 0]
        num_frame = len(x_gt)
        n_dim = seq_len
        total_error = (error.pow(2).sum() + init_error.pow(2).sum()) / (num_frame * n_dim)
        return total_error

    model = VAEModel(out_dim=1, con_dim=3, sqz_prior=True, cond_drop=False, use_prior_flow=True, device=self.device)
    model.load_state_dict(torch.load('VAEflow.ckpt'))
    for param in model.parameters():
        param.requires_grad = False

    model_out = {}
    sample_out = {}
    csv_dataset = CSVDataset('/root/tzy/VAE_rnn/AMP_VAEflow/datasets/combined_dataset_with_point.csv')
    data_loader = DataLoader(csv_dataset, batch_size=32, shuffle=True)
    t = 0
    for idx, batch in enumerate(data_loader):
        out, _, _ = model(batch=batch, ret=model_out, train=False, return_latent=False, temperature=1.)
        sample_out.append(out)
        t+=1
        if t==1000:
            break

    print(sample_out)

    csv_dataset.close()

