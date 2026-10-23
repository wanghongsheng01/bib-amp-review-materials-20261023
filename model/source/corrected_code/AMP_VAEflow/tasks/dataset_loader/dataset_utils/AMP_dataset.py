import torch
import numpy as np
import pickle as pkl
import os, sys
import math, random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tqdm
from utils.commons.indexed_datasets import IndexedDataset
class LRS3SeqDataset(Dataset):
    def __init__(self, prefix='train', protein_file=None, chemical_file=None):
        if protein_file and chemical_file:
            with open(protein_file, 'r') as file:
                self.protein_sequences = file.read().splitlines()
            # seq2seq
            self.chemical_descriptors = pd.read_csv(chemical_file, delimiter='/n', header=None).values
            self.chemical_descriptors = self.chemical_descriptors.flatten()
            self.chemical_descriptors = self.chemical_descriptors.tolist()
            assert len(self.protein_sequences) == len(self.chemical_descriptors), \
                "The number of protein sequences must match the number of chemical descriptors."



    def _get_item(self, index):
        """
        This func is necessary to open files in multi-threads!
        """
        if self.ds is None:
            self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        return self.ds[index]


    def __getitem__(self, idx):
        # if hparams['load_db_to_memory']:
        #     return self.memory_cache[idx]

        raw_item = self._get_item(idx)
        if raw_item is None:
            print("loading from binary data failed!")
            return None
        item = {}
        item_id = raw_item['item_id']  # str: "<speakerID>_<clipID>"
        item['item_id'] = item_id
        # audio-related features
        mel = raw_item['mel']
    def __getitem__(self, idx):
        # item = super().__getitem__(idx)  # 调用原有的__getitem__方法获取数据

        if hasattr(self, 'protein_sequences') and hasattr(self, 'chemical_descriptors'):
            protein_seq = self.protein_sequences[idx]
            chem_desc = self.chemical_descriptors[idx]

            # 转换蛋白质序列为张量（根据您的模型需要做相应的处理）
            protein_tensor = torch.tensor([ord(c) for c in protein_seq], dtype=torch.long)
            chem_tensor = torch.tensor(chem_desc, dtype=torch.float)

            item['protein_sequence'] = protein_tensor
            item['chemical_descriptor'] = chem_tensor

        return item

    def collater(self, samples):
        batch = super().collater(samples)  # 调用原有的collater方法获取批量数据

        if 'protein_sequence' in samples[0]:
            protein_seqs = [s['protein_sequence'] for s in samples]
            chem_descs = [s['chemical_descriptor'] for s in samples]
            batch['protein_seqs'] = torch.nn.utils.rnn.pad_sequence(protein_seqs, batch_first=True, padding_value=0)
            batch['chem_descs'] = torch.stack(chem_descs, dim=0)

        return batch

    def get_dataloader(self):
        shuffle = True if self.db_key == 'train' else False
        max_tokens = 60000
        batches_idx = self.batch_by_size(self.ordered_indices(), max_tokens=max_tokens)
        batches_idx = batches_idx * 50
        random.shuffle(batches_idx)
        loader = DataLoader(self, pin_memory=True, collate_fn=self.collater, batch_sampler=batches_idx, num_workers=4)
        return loader

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1):
        """
        Yield mini-batches of indices bucketed by size. Batches may contain
        sequences of different lengths.

        Args:
            indices (List[int]): ordered list of dataset indices
            num_tokens_fn (callable): function that returns the number of tokens at
                a given index
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
        """
        def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            if len(batch) == 0:
                return 0
            if len(batch) == max_sentences:
                return 1
            if num_tokens > max_tokens:
                return 1
            return 0

        num_tokens_fn = lambda x: self.sizes[x]
        max_tokens = max_tokens if max_tokens is not None else 60000
        max_sentences = max_sentences if max_sentences is not None else 512
        bsz_mult = required_batch_size_multiple

        sample_len = 0
        sample_lens = []
        batch = []
        batches = []
        for i in range(len(indices)):
            idx = indices[i]
            num_tokens = num_tokens_fn(idx)
            sample_lens.append(num_tokens)
            sample_len = max(sample_len, num_tokens)

            assert sample_len <= max_tokens, (
                "sentence at index {} of size {} exceeds max_tokens "
                "limit of {}!".format(idx, sample_len, max_tokens)
            )
            num_tokens = (len(batch) + 1) * sample_len

            if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
                mod_len = max(
                    bsz_mult * (len(batch) // bsz_mult),
                    len(batch) % bsz_mult,
                )
                batches.append(batch[:mod_len])
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
            batch.append(idx)
        if len(batch) > 0:
            batches.append(batch)
        return batches


if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    ds = LRS3SeqDataset('train',protein_file='/Users/tang/PycharmProjects/AMP_VAEflow/datasets/datasets_for_VAE.txt',
                        chemical_file='/Users/tang/PycharmProjects/AMP_VAEflow/datasets/output_seq.csv')
    ret = ds[0]
    loader = ds.get_dataloader()
    pbar = tqdm.tqdm(total=len(ds.batch_by_size(ds.ordered_indices())))
    # for i in tqdm.trange(len(ds)):
    #     ds[i]
    for batch in loader:
        pbar.update(1)

    pbar = tqdm.tqdm(total=len(ds.batch_by_size(ds.ordered_indices())))
    for batch in loader:
        pbar.update(1)
    a=1
