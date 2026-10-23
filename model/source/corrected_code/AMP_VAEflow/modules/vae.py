from tasks.dataset_loader.dataset_utils.amp_input_contract import parse_model_batch
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
import numpy as np
from modules.flow_base import WN, ResidualCouplingBlock
from modules.attention_model import AttentionFusion
# from modules.transformer_base import Embedding
PAD_IDX = 20


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
            torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e4)  # an arbitrary large number


# encoder
class FVAEEncoder(nn.Module):
    # 输入有两个：1.x[bs,emb_dim,seq_len];2.g_sqz[bs,emb_dim,seq_len]
    def __init__(self, hidden_channels, latent_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0, strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.att = AttentionFusion(feature_dim=hidden_channels, feature_dim2=9)
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels * 2, 1)
        self.latent_channels = latent_channels


    def forward(self, x, g, h):

        x = self.wn(x, g)
        if h!=None:

            x = self.att(sequence=x.permute(0, 2, 1).to(torch.float), feature=h.to(torch.float))

        x = self.out_proj(x)

        m, logs = torch.split(x, self.latent_channels, dim=1)

        z = (m + torch.randn_like(m) * torch.exp(logs))

        return z, m, logs



class FVAEDecoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0,
                 strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.wn = WN(latent_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(latent_channels, out_channels, 1)


    def forward(self, x):
        g = None
        x = self.wn(x, g)
        x2 = self.out_proj(x)
        return x2.squeeze(1)


# 潜在空间维度只有16
class FVAE(nn.Module):
    def __init__(self, out_channels,
                 hidden_channels=256, latent_size=16,
                 kernel_size=3, enc_n_layers=5, dec_n_layers=5, gin_channels=16, strides=[2],
                 use_prior_glow=True, glow_hidden=256, glow_kernel_size=3, glow_n_blocks=5,
                 sqz_prior=False, use_pos_emb=False, n_vocab=21, emb_dim=100, PAD_IDX=PAD_IDX):
        super(FVAE, self).__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.latent_size = latent_size
        self.use_prior_glow = use_prior_glow
        self.sqz_prior = sqz_prior




        self.g_pre_net = nn.Conv1d(in_channels=gin_channels, out_channels=emb_dim, kernel_size=3, padding=1)


        self.word_emb = nn.Embedding(n_vocab, emb_dim, PAD_IDX)


        self.encoder = FVAEEncoder(emb_dim, latent_size, kernel_size,
                                   enc_n_layers, gin_channels=emb_dim, strides=strides)
        if use_prior_glow:

            self.prior_flow = ResidualCouplingBlock(
                latent_size, glow_hidden, glow_kernel_size, 1, glow_n_blocks, 4, gin_channels=gin_channels)
        self.use_pos_embed = use_pos_emb

        if sqz_prior:
            self.query_proj = nn.Linear(latent_size, latent_size)
            self.key_proj = nn.Linear(latent_size, latent_size)
            self.value_proj = nn.Linear(latent_size, hidden_channels)
            self.decoder = FVAEDecoder(hidden_channels, hidden_channels, out_channels, kernel_size,
                                       dec_n_layers, gin_channels, strides=strides)

            if self.use_pos_embed:
                self.embed_positions = SinusoidalPositionalEmbedding(self.latent_size, 0, init_size=2000 + 1, )
        else:
            self.decoder = FVAEDecoder(latent_size, hidden_channels, out_channels, kernel_size,
                                       dec_n_layers, gin_channels, strides=strides)

        self.prior_dist = dist.Normal(0, 1)

    # decoder
    def infer(self, z):
        # z[1,16,seq_len]
        if self.sqz_prior:

            if self.use_pos_embed:
                position = self.embed_positions(z.transpose(1, 2).abs().sum(-1)).transpose(1, 2)
                z = z + position
            q = self.query_proj(z.mean(dim=-1, keepdim=True).transpose(1, 2))  # [B, 1, C=16]
            k = self.key_proj(z.transpose(1, 2))  # [B, T, C=16]
            v = self.value_proj(z.transpose(1, 2))  # [B, T, C=256]
            attn = torch.bmm(q, k.transpose(1, 2))  # [B, 1, T]
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(attn, v)  # [B, 1, C=256]
            style_encoding = out.repeat([1, z.shape[-1], 1]).transpose(1, 2)  # [B, C=256, T]
            x_recon = self.decoder(style_encoding)

        else:
            x_recon = self.decoder(z)

        x_recon = torch.clamp(x_recon, min=0, max=20)

        decoded_indices = torch.round(x_recon).long()  # [B, T]


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

        index_to_amino_acid = {index: aa for aa, index in amino_acid_to_index.items()}

        def decode_sequences(decoded_indices, index_to_amino_acid, pad_index):
            amino_acid_seq = []
            out = ''
            for idx in decoded_indices[0]:
                if int(idx) == pad_index:
                    break
                amino_acid_seq.append(index_to_amino_acid[int(idx)])
                out = ''.join(amino_acid_seq)
            return out

        decoded_amino_acid_sequences = decode_sequences(decoded_indices, index_to_amino_acid, PAD_IDX)

        return decoded_amino_acid_sequences

    def forward(self, x=None, g=None, h=None, infer=False, temperature=1., **kwargs):
        """
        :param x: [T, B]
        :param g: [T, N1, B]
        :param h: [N2, B]
        :return:
        """

        x = self.word_emb(x).permute(1, 2, 0)  # [B, C, T]
        g_for_sqz = g.permute(2, 1, 0).float()
        # [B,C,T]
        g_sqz = self.g_pre_net(g_for_sqz)

        if not infer:
            # Inputs is embeddings of: mbsize  x emb_dim x seq_len
            z_q, m_q, logs_q = self.encoder(x, g_sqz, h)
            if self.sqz_prior:
                z = z_q
                if self.use_pos_embed:
                    position = self.embed_positions(z.transpose(1, 2).abs().sum(-1)).transpose(1, 2)
                    z = z + position
                q = self.query_proj(z.mean(dim=-1, keepdim=True).transpose(1, 2))  # [B, 1, C=16]
                k = self.key_proj(z.transpose(1, 2))  # [B, T, C=16]
                v = self.value_proj(z.transpose(1, 2))  # [B, T, C=256]
                attn = torch.bmm(q, k.transpose(1, 2))  # [B, 1, T]
                attn = F.softmax(attn, dim=-1)
                out = torch.bmm(attn, v)  # [B, 1, C=256]
                style_encoding = out.repeat([1, z_q.shape[-1], 1]).transpose(1, 2)  # [B, C=256, T]
                x_recon = self.decoder(style_encoding)


            else:
                x_recon = self.decoder(z_q)
            q_dist = dist.Normal(m_q, logs_q.exp())
            if self.use_prior_glow:
                logqx = q_dist.log_prob(z_q)
                z_p = self.prior_flow(z_q, g_for_sqz)
                logpx = self.prior_dist.log_prob(z_p)
                loss_kl = (logqx - logpx).sum() / logqx.shape[1]

            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
                loss_kl = (loss_kl).sum() / z_q.shape[1]
                z_p = z_q
            return x_recon, loss_kl, z_p.transpose(1, 2), m_q.transpose(1, 2), logs_q.transpose(1, 2)


        else:
            z_q, m_q, logs_q = self.encoder(x, g_sqz, h)
            if self.sqz_prior:
                z = z_q
                if self.use_pos_embed:
                    position = self.embed_positions(z.transpose(1, 2).abs().sum(-1)).transpose(1, 2)
                    z = z + position
                q = self.query_proj(z.mean(dim=-1, keepdim=True).transpose(1, 2))  # [B, 1, C=16]
                k = self.key_proj(z.transpose(1, 2))  # [B, T, C=16]
                v = self.value_proj(z.transpose(1, 2))  # [B, T, C=256]
                attn = torch.bmm(q, k.transpose(1, 2))  # [B, 1, T]
                attn = F.softmax(attn, dim=-1)
                out = torch.bmm(attn, v)  # [B, 1, C=256]
                style_encoding = out.repeat([1, z_q.shape[-1], 1]).transpose(1, 2)  # [B, C=256, T]
                x_recon = self.decoder(style_encoding)


            else:
                x_recon = self.decoder(z_q)

            if self.use_prior_glow:
                z_p = self.prior_flow(z_q, g_for_sqz)

            return x_recon, z_p.transpose(1, 2), m_q.transpose(1, 2), logs_q.transpose(1, 2)


class VAEModel(nn.Module):
    def __init__(self, out_dim=1, con_dim=3, latent_size=16, sqz_prior=False, cond_drop=False, use_prior_flow=True,
                 device='cpu'):
        super().__init__()
        self._device = device
        self.cond_drop = cond_drop
        if self.cond_drop:
            self.dropout = nn.Dropout(0.5)

        self.out_dim = out_dim
        self.sqz_prior = sqz_prior
        self.use_prior_flow = use_prior_flow
        self.vae = FVAE(out_channels=self.out_dim, hidden_channels=256, latent_size=latent_size, kernel_size=5,
                        enc_n_layers=8, dec_n_layers=4, gin_channels=con_dim, strides=[1],
                        use_prior_glow=self.use_prior_flow, glow_hidden=64, glow_kernel_size=3,
                        glow_n_blocks=4, sqz_prior=sqz_prior).to(self._device)
        # self.vae=torch.nn.DataParallel(self.vae, device_ids=[0, 1, 2, 3])

        self.downsampler = LambdaLayer(
            lambda x: F.interpolate(x.transpose(1, 2), scale_factor=0.5, mode='nearest').transpose(1, 2))

    def num_params(self, model, print_out=True, model_name="modules"):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters

    @property
    def device(self):
        return self.vae.parameters().__next__().device

    # Inputs is batch of sentences: seq_len x mbsize
    def forward(self, batch, ret, train=True, return_latent=False, temperature=1.):
        infer = not train
        # Strict numeric parsing; rejects incomplete, nonfinite or misaligned data.
        # Historical abs behavior is retained explicitly, not scientifically endorsed.
        parsed_data, batch_parsed = parse_model_batch(batch)
        numpy_data = np.array(parsed_data, dtype=float)
        point_describe = torch.abs(torch.tensor(numpy_data).to(self._device))

        max_units = 48
        max_subunits = max(len(unit) for item in batch_parsed for unit in item)
        batch_tensor = torch.zeros((len(batch_parsed), max_units, max_subunits))
        

        for i, item in enumerate(batch_parsed):
            for j, unit in enumerate(item):
                batch_tensor[i, j, :len(unit)] = torch.tensor(unit)
        # [B,T,N]-->[T,N,B]
        describe = batch_tensor.to(self._device).permute(1, 2, 0)


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
        indexed_sequences = [[amino_acid_to_index[aa] for aa in seq] for seq in batch['protein']]

        pad_index = PAD_IDX
        max_length = 48
        padded_sequences = [seq + [pad_index] * (max_length - len(seq)) for seq in indexed_sequences]
        batch_sequences = torch.tensor(padded_sequences)

        if not infer:
            # [B,T]-->[T, B]
            exp = batch_sequences.to(self.device).permute(1, 0)
            x = exp
            x_recon, loss_kl, z_p, m_q, logs_q = self.vae(x=x, g=describe, h=point_describe, infer=False)
            ret['pred'] = x_recon
            ret['mask'] = (batch_sequences != PAD_IDX).float()
            ret['loss_kl'] = loss_kl
            if return_latent:
                ret['m_q'] = m_q
                ret['z_p'] = z_p
            return x_recon, loss_kl, m_q, logs_q
        else:
            exp = batch_sequences.to(self.device).permute(1, 0)
            x_recon, z_p, m_q, logs_q = self.vae(x=exp, g=describe, h=point_describe, infer=True, temperature=temperature)
            return z_p, m_q, logs_q


if __name__ == '__main__':
    model = FVAE(out_channels=1, hidden_channels=128, latent_size=32, kernel_size=3, enc_n_layers=6, dec_n_layers=2,
                 gin_channels=3, strides=[1], glow_hidden=128, glow_kernel_size=3, glow_n_blocks=3,
                 n_vocab=20, emb_dim=100, PAD_IDX=1, use_pos_emb=True, sqz_prior=True, use_prior_glow=True)

    # Inputs x is batch of sentences: seq_len x mbsize
    x = torch.randint(0, 20, ([30, 8]))
    # Inputs g is batch of 描述符 of sentences: seq_len x mbsize
    g = torch.randint(0, 20, ([30, 3, 8]))
    train_out = model(x, g, infer=False)
    x_recon, loss_kl, z_p, m_q, logs_q = train_out
    print("train ok")
    infer_out = model(x, g, infer=True)
    x_recon, z_p = infer_out
    print("infer ok")

