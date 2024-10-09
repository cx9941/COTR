import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='en', type=str)
parser.add_argument('--llm_type', default='gpt', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--turn', default=0, type=int)
args = parser.parse_args()

args.input_dir_name = f"seed{args.seed}"

args.output_dir = f"outputs/{args.dataset_name}/{args.llm_type}/{args.input_dir_name}/input{args.turn}"
args.result_dir = f"results/{args.dataset_name}/{args.llm_type}/{args.input_dir_name}"
args.next_input_path = f"data/{args.dataset_name}/{args.llm_type}/{args.input_dir_name}/input{args.turn + 1}.csv"
args.input_path = f"data/{args.dataset_name}/{args.llm_type}/{args.input_dir_name}/input{args.turn}.csv"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)