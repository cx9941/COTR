import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='eu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--turn', type=int, default=2)
args = parser.parse_args()

args.input_dir_name = f"seed{args.seed}"

args.output_dir = f"outputs/{args.dataset_name}/{args.input_dir_name}/input{args.turn}"
args.result_dir = f"results/{args.dataset_name}/{args.input_dir_name}"
args.next_input_dir = f"data/{args.dataset_name}/{args.input_dir_name}/input{args.turn + 1}"
args.input_dir = f"data/{args.dataset_name}/{args.input_dir_name}/input{args.turn}"

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

if not os.path.exists(args.next_input_dir):
    try:
        os.makedirs(args.next_input_dir)
    except Exception as e:
        print(e)
    