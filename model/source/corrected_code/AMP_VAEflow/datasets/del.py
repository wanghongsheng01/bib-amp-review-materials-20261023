import csv
import os
input_csv_file = 'output_seq.csv'
output_csv_file = 'seq_describe.csv'

# 打开输入文件和输出文件
with open(input_csv_file, 'r', newline='') as infile, open(output_csv_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

    # 逐行读取并写入，去除双引号
    for row in reader:
        # 去除每个字段值两边的双引号
        row = [field.strip('"') for field in row]
        # 检查字段中是否存在分隔符（如果是逗号分隔的CSV，则检查逗号）
        row = [field.replace(',', '') for field in row]
        writer.writerow(row)

# 删除输入文件
os.remove(input_csv_file)