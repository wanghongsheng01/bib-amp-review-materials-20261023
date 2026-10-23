# 本文件用于生成随便一个氨基酸序列的化学描述符
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import csv

# 天然氨基酸的对照表
# 字典，将氨基酸的单字母代码转换为它们的SMILES表示
# amino_acid_smiles = {
#     'A': 'CC(C(=O)O)N',
#     'R': 'NC(CCCNC(=N)N)C(=O)O',
#     'N': 'NC(CC(=O)N)C(=O)O',
#     'D': 'NC(CCC(=O)O)C(=O)O',
#     'C': 'N[C@@H](CS)C(=O)O',
#     'Q': 'N[C@@H](CCC(=O)N)C(=O)O',
#     'E': 'N[C@@H](CCC(=O)O)C(=O)O',
#     'G': 'C(C(=O)O)N',
#     'H': 'N[C@@H](CC1=CN=CN1)C(=O)O',
#     'I': 'CC[C@H](C)[C@@H](C(=O)O)N',
#     'L': 'CC(C)C[C@@H](C(=O)O)N',
#     'K': 'N[C@@H](CCCCN)C(=O)O',
#     'M': 'CSCC[C@@H](C(=O)O)N',
#     'F': 'N[C@@H](CC1=CC=CC=C1)C(=O)O',
#     'P': 'O=C1CCCN1C(=O)O',
#     'S': 'N[C@@H](CO)C(=O)O',
#     'T': 'CC(C)[C@@H](O)C(=O)O',
#     'W': 'N[C@@H](CC1=CNC2=C1C=CC=C2)C(=O)O',
#     'Y': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)O',
#     'V': 'CC(C)[C@@H](C(=O)O)N'
# }
#
# # smiles_string = 'CCO'
# m = Chem.MolFromSmiles('CC[C@H](C)[C@@H](C(=O)O)N')
# print(m)
# # Draw.MolToImage(m)
# # mol1 = Chem.MolFromSmiles('CCO')
# # mol2 = Chem.MolFromSmiles('CCN')
# # fp1 = FingerprintMols.FingerprintMol(mol1)
# # fp2 = FingerprintMols.FingerprintMol(mol2)
# # similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
# # print("相似性分数：", similarity)
# # mol_weight = Descriptors.MolWt(m)
# # print("分子量：", mol_weight)
#
# # 一个示例氨基酸序列
# sequence = 'IAGKSESLWAGK'
# # 将氨基酸序列转换为SMILES列表
# molecule = Chem.MolFromSequence(sequence)
# mol_weight = Descriptors.MolWt(m)
# # 获取多肽的SMILES
# peptide_smiles = Chem.MolToSmiles(molecule)
# print(sequence, mol_weight, peptide_smiles)

class seq2rdkit:
    def __init__(self):
        pass

    def calculate_descriptors(self, aa):
        mol = Chem.MolFromSequence(aa)
        # 分子模量
        a = Descriptors.MolWt(mol)
        a = round(a, 2)
        # 分配系数
        b = Descriptors.MolLogP(mol)
        b = round(b, 2)
        # 拓扑极性表面积
        c = Descriptors.TPSA(mol)
        c = round(c, 2)

        return [a, b, c]

    # 按照原序列输出每个氨基酸的描述符
    def sequence_descriptors(self, sequence):
        descriptors_list = []
        for aa in sequence:
            desc = self.calculate_descriptors(aa)
            descriptors_list.append(desc)
        return descriptors_list

    def seq2seq(self, sequence):
        # 脚本1:一个蛋白质序列，转换成每个氨基酸的分子描述符的序列
        list = self.sequence_descriptors(sequence)
        return list

    def seq2point(self, sequence):
        # 脚本2:一个蛋白质序列，转换成整个多肽的分子描述符
        list = self.calculate_descriptors(sequence)
        return list


if __name__ == '__main__':
    # 一个示例氨基酸序列
    # sequence = 'IAGKSESLWAGK'
    # # 将氨基酸序列转换为SMILES列表
    # molecule = Chem.MolFromSequence(sequence)
    # # 获取多肽的SMILES
    # peptide_smiles = Chem.MolToSmiles(molecule)
    # print(sequence, peptide_smiles)

  

    import csv

    # 假设 seq2rdkit 是一个你已经定义的类
    cla = seq2rdkit()

    # 读取、处理和写入数据，而不将其存储在内存中
    with open('datasets_for_VAE.txt', 'r') as infile, \
            open('output_seq.csv', 'w', newline='') as seqfile:

        seq_writer = csv.writer(seqfile, quoting=csv.QUOTE_MINIMAL)
        # point_writer = csv.writer(pointfile, quoting=csv.QUOTE_MINIMAL)

        for i, line in enumerate(infile):
            print(f'Processing line {i}')
            sequence = line.strip()
            result1 = cla.seq2seq(sequence)
            # result2 = cla.seq2point(sequence)

            # 直接写入文件
            seq_writer.writerow(result1)
            # point_writer.writerow(cleaned_result2)

    print('CSV文件已写入')
