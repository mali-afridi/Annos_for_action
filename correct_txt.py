import os
import argparse
parser= argparse.ArgumentParser()
parser.add_argument('text_file',type=str)
args = parser.parse_args()
# Input video path
correct = []
text_file = args.text_file
f = open(text_file)
data = f.readlines()
f.close()
for action in data:
    correct.append(action.strip().split('\t')[1])

text_file_name = text_file.split('.')[0]
with open(f'{text_file_name}_correct.txt','a') as f:
    for action in correct:
        f.write(action+'\n')
f.close()
print("done")