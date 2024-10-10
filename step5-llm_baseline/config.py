import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='eu')
parser.add_argument('--backbone', type=str, default='Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

args.output_dir = f"outputs/{args.dataset_name}/seed{args.seed}"
args.result_dir = f"results/{args.dataset_name}/seed{args.seed}"
args.input_path = f"data/{args.dataset_name}.csv"

if not os.path.exists(args.output_dir):
    try:
        os.makedirs(args.output_dir)
    except Exception as e:
        print(e)

if not os.path.exists(args.result_dir):
    try:
        os.makedirs(args.result_dir)
    except Exception as e:
        print(e)
    