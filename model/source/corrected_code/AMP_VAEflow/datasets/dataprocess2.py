from Bio import SeqIO

input_fasta = 'uniprot_trembl.fasta'
valid_sequences = 'datasets_for_VAE.txt'
valid_aa_set = set('ACDEFGHIKLMNPQRSTVWY')

# Function to check if sequence is valid and shorter than 200
def is_valid_and_short_sequence(record):
    sequence = str(record.seq)
    if len(sequence) > 48:
        return False
    if set(sequence) <= valid_aa_set:
        return True
    return False

if __name__ == "__main__":
    # Open the output file
    with open(valid_sequences, 'w') as file:
        # Open the input FASTA file and process it line by line
        for record in SeqIO.parse(input_fasta, 'fasta'):
            if is_valid_and_short_sequence(record):
                file.write(str(record.seq) + '\n')

    print(f"Processing complete, filtered protein sequences have been saved to {valid_sequences}")
