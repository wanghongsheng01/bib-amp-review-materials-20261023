# 该脚本用于从原始的fasta数据集中提取纯蛋白数据文本，并筛选数据重建纯蛋白数据集
from Bio import SeqIO
from rdkit import Chem

#
# 输入FASTA文件的路径
input_fasta = 'uniprot_trembl.fasta'
# 输出纯序列的文本文件路径
output_txt = 'unreviewed_sequences.txt'

# 打开输出文件准备写入
with open(output_txt, 'w') as out_file:
    # 使用BioPython的SeqIO.parse读取FASTA文件
    for record in SeqIO.parse(input_fasta, 'fasta'):
        # 将序列转换为字符串并写入文件，每个序列占一行
        out_file.write(str(record.seq) + '\n')


print(f"处理完成，提取的蛋白质序列已保存到 {output_txt}")

# 数据预处理，删除无法建模的序列
with open(f'/Users/tang/PycharmProjects/AMP_VAEflow/datasets/{output_txt}', 'r') as file:
    for line in file:
        # 移除行尾的换行符
        sequence = line.strip()
        mol = Chem.MolFromSequence(sequence)
        if mol is not None:
            with open('/Users/tang/PycharmProjects/AMP_VAEflow/datasets/datasets_for_VAE.txt', 'a') as file1:
                file1.write(line)
print(f"处理完成，筛选后的蛋白质序列已保存到 datasets_for_VAE.txt")