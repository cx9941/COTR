import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='jp')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
print(args)