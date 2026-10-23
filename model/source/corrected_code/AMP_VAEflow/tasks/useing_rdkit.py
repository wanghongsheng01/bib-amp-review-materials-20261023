
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import csv


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

class seq2rdkit:
    def __init__(self):
        pass

    def calculate_descriptors(self, aa):
        mol = Chem.MolFromSequence(aa)
        a = Descriptors.MolWt(mol)
        b = Descriptors.NumHAcceptors(mol)
        c = Descriptors.NumHDonors(mol)

        return [a, b, c]

    def sequence_descriptors(self, sequence):
        descriptors_list = []
        for aa in sequence:
            desc = self.calculate_descriptors(aa)
            descriptors_list.append(desc)
        return descriptors_list

    def seq2seq(self, sequence):
        list = self.sequence_descriptors(sequence)
        return list

    def seq2point(self, sequence):
        list = self.calculate_descriptors(sequence)
        return list


if __name__ == '__main__':
   
    cla = seq2rdkit()
    templist=[]
    templist2=[]
    with open('/Users/tang/PycharmProjects/AMP_VAEflow/datasets/datasets_for_VAE.txt', 'r') as file:
        for i, line in enumerate(file):
            print(f'line{i}')
            sequence = line.strip()
            result1 = cla.seq2seq(sequence)
            result2 = cla.seq2point(sequence)
            templist.append(result1)
            templist2.append(result2)


    with open('/Users/tang/PycharmProjects/AMP_VAEflow/datasets/output_seq.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(templist)

    with open('/Users/tang/PycharmProjects/AMP_VAEflow/datasets/output_point.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(templist2)
