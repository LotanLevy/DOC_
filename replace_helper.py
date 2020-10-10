
import argparse
import os


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--file_path', '-fp', type=str, required=True )
parser.add_argument('--src', type=str, required=True )
parser.add_argument('--dest', '-o', type=str, required=True)

args = parser.parse_args()

new_lines = []

with open(args.file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        rel = os.path.relpath(line, args.src)
        print(rel)
        new_lines.append(os.path.join(args.dest, rel))

with open(args.file_path, 'w') as f:
    f.writelines(new_lines)
