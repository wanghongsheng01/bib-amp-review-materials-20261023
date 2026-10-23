import csv

# 假设你的文件名分别为 'proteins.txt' 和 'descriptions.csv'
protein_file = 'datasets_for_VAE.txt'
description_file = 'seq_describe.csv'
output_csv_file = 'datasets_vae.csv'

# 打开三个文件：蛋白质序列文件，描述文件，以及输出文件
with open(protein_file, 'r') as pf, open(description_file, 'r') as df, open(output_csv_file, 'w', newline='') as of:
    protein_reader = (line.strip() for line in pf)  # 创建一个生成器，逐行读取蛋白质序列
    description_reader = (line.strip() for line in df)  # 创建一个生成器，逐行读取描述

    csv_writer = csv.writer(of)
    csv_writer.writerow(['idx', 'protein', 'describe'])  # 写入标题行

    # 使用enumerate逐行合并数据，并写入输出文件
    for idx, (protein, description) in enumerate(zip(protein_reader, description_reader), start=1):
        print(f"{idx},done")
        csv_writer.writerow([idx, protein, description])

# 文件会在with语句块结束时自动关闭
