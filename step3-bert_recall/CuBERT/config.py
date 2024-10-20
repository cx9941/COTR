import argparse
import os
import logging

def get_args():
    parser = argparse.ArgumentParser(description='BERT Text Matching')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--top_num', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length for BERT')
    parser.add_argument('--k', type=int, default=10, help='Maximum sequence length for BERT')
    parser.add_argument('--train_file_path', type=str, default='../data/task_data.csv', help='Pretrained BERT model')
    parser.add_argument('--dataset_name', type=str, default='en', help='Device to run the model on')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--mode', type=str, default='train', help='Device to run the model on')
    parser.add_argument('--bert_mode', type=str, default='normal', help='Device to run the model on')
    args = parser.parse_args()
    args.checkpoint_path = f"outputs/{args.dataset_name}/{args.learning_rate}_{args.batch_size}_{args.train_ratio}"
    args.result_path = f"results/{args.dataset_name}/{args.learning_rate}_{args.batch_size}_{args.train_ratio}"
    if args.dataset_name == 'jp':
        args.bert_model = '../../llms/bert-base-japanese'
    else:
        args.bert_model = '../../llms/bert-base-uncased'
        
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    # 配置日志
    logging.basicConfig(level=logging.DEBUG,  # 设置日志级别
                        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
                        filename=f'{args.checkpoint_path}/run_{args.mode}.log',  # 输出到文件
                        filemode='w')  # 写模式，如果文件存在会覆盖
    return args