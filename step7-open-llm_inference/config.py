import os
import argparse
from utils import get_lastest_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='en')
parser.add_argument('--mode', type=str, default='all')
parser.add_argument('--model_name', type=str, default='nanbeige-sft')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--thred', type=int, default=50)
parser.add_argument('--mapping',default='df', choices=['le', 'df'],type=str, help='')
args = parser.parse_args()

args.data_path = f'data/{args.dataset_name}.csv'
args.label_path = f'../step3-bert_recall/data/{args.dataset_name}/task.csv'


map_dir = {
    "llama":  f"../llms/Meta-Llama-3.1-8B-Instruct",
    "nanbeige":  f"../llms/Nanbeige2-8B-Chat",
    "baichuan":  f"../llms/Baichuan2-7B-Chat",
    "baichuan-sft":  f"../step6-open_llm_sft/outputs/{args.dataset_name}/baichuan/sft",
    "nanbeige-sft":  f"../step6-open_llm_sft/outputs/{args.dataset_name}/nanbeige/sft",
    "llama-sft":  f"../step6-open_llm_sft/outputs/{args.dataset_name}/llama/sft",
    "baichuan-sft-enhanced":  f"../step6-open_llm_sft/outputs/{args.dataset_name}-enhanced/baichuan/sft",
    "nanbeige-sft-enhanced":  f"../step6-open_llm_sft/outputs/{args.dataset_name}-enhanced/nanbeige/sft",
    "llama-sft-enhanced":  f"../step6-open_llm_sft/outputs/{args.dataset_name}-enhanced/llama/sft",
}

args.model_path = map_dir[args.model_name]
if 'sft' in args.model_name:
    args.model_path = get_lastest_checkpoint(args.model_path)

if args.mode == 'all':
    args.output_dir = f"outputs_all/{args.dataset_name}/{args.model_name}/seed{args.seed}"
    args.metrics_dir = f"metrics_all/{args.dataset_name}/{args.model_name}/seed{args.seed}"
else:
    args.output_dir = f"outputs/{args.dataset_name}/{args.model_name}/seed{args.seed}"
    args.metrics_dir = f"metrics/{args.dataset_name}/{args.model_name}/seed{args.seed}"
args.input_path = f"data/{args.dataset_name}.csv"
args.metric_path = f'{args.metrics_dir}/thred_{args.thred}.json'

if not os.path.exists(args.output_dir):
    try:
        os.makedirs(args.output_dir)
    except Exception as e:
        print(e)
        
if not os.path.exists(args.metrics_dir):
    try:
        os.makedirs(args.metrics_dir)
    except Exception as e:
        print(e)