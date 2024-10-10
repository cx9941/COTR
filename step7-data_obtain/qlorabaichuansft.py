from datasets import load_dataset
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import re
import random
from tqdm import tqdm
import os
import argparse
import json
from tqdm import tqdm
import time
from datasets import load_from_disk
import pandas as pd
import os
from torch.utils.data import DataLoader
import re
import torch
import transformers
import random
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    TrainingArguments,
    BitsAndBytesConfig
)
import bitsandbytes as bnb
import numpy as np
from datasets import Dataset
from torch.nn.functional import softmax
from sklearn.metrics import ndcg_score
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# device = torch.device("cuda:1" if torch.cuda.is_available() else"cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',default="/home/data/qinchuan/TMIS/paper_code/llm_models/baichuan-inc/Baichuan2-13B-Base",type=str,help='')
    # parser.add_argument('--model_dir',default="public/baichuan-inc/Baichuan-13B-Base",type=str,help=')
    parser.add_argument('--output_dir',default='/home/data/qinchuan/TMIS/paper_code/output/data_obtain/baichuan_qlora',type=str,help='')
    parser.add_argument('--num_train_epochs',default=3,type=int,help='')
    parser.add_argument('--lr',default=3e-5,type=float,help='')
    parser.add_argument('--per_device_train_batch_size',default=1,type=int,help='')
    parser.add_argument('--gradient_accumulation_steps',default=8,type=int,help='')
    parser.add_argument('--local_rank',type=int,default=0,help='')
    parser.add_argument('--bits', type=int, default=4, help='')
    parser.add_argument('--double_quant', type=bool, default=True, help="Compress the quantization statistics through double quantization.")
    # parser.add_argument('-lora_r',type=int,default=8,help='')
    parser.add_argument("--deepspeed",type=str,default='/home/data/qinchuan/TMIS/paper_code/data_obtain/ds_zero2.json',help="path to deepspeed")
    return parser.parse_args()

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(args, checkpoint_dir):
    n_gpus = 1
    max_memory = f'80000MB'
    max_memory = {i + 4: max_memory for i in range(n_gpus)}

    # if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_dir}...')
    compute_dtype = torch.bfloat16
    # compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type='nf4'
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('=' * 80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    modules = find_all_linear_names(args, model)

    # 会自动计算应用lora的地方
    # if args.model_name == "chatglm":
    #     modules = ["query_key_value"]

    model.config.torch_dtype = torch.bfloat16

    # if not args.full_finetune:
    model = prepare_model_for_kbit_training(model)
    # if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, 'adapter_model'))
        for name, p in model.named_parameters():
            if 'lora' in name:
                print(name, p.sum())
    else:
        print(f'adding LoRA modules...')
        model = get_peft_model(model, config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            # if args.bf16:
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                # if args.bf16 and module.weight.dtype == torch.float32:
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def main():
    args = set_args()
    # model = AutoModelForCausalLM.from_pretrained(args.model_dir,trust_remote_code=True)
    model = get_accelerate_model(args, None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     target_modules=["q_proj", "v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # model = get_peft_model(model, peft_config)

    with open("/home/data/qinchuan/TMIS/paper_code/banking_data/rank.json", 'r') as f:
        data = json.load(f)
    # result = pd.read_csv('')
    # query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/banking77.csv')

    # text_list =list(query['text'])
    # query_list= []
    # for i in result['text']:
    #     index=text_list.index(i)
    #     text = query.iloc[index]['text']
    #     name = "".join([f"class{j}." + query.iloc[index][f"top{j}_name"] for j in range(1, 78)])
    #     q = f'Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:{name}\n,specific text:{text}\n please sort the above 77 types of intents according to the degree of matching with the intent of the text. \n Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n.'
    #     query_list.append(q)


    with open("/home/data/qinchuan/TMIS/paper_code/banking_data/categories.json", 'r') as f:
        labellist = json.load(f)
    result = pd.read_csv("/home/data/qinchuan/TMIS/paper_code/banking_data/train.csv")

    candidate = labellist[:40]
    tmpnum = {}
    for i in candidate:
        tmpnum[i] = 0

    text_new = []
    label_new = []

    query_list= []
    for index in range(len(data)):
        text = data[index]['text']
        name = "".join([f"class{j}." + data[index][f"top{j}"] for j in range(1, 78)])
        q = f'Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:{name}\n,specific text:{text}\n please sort the above 77 types of intents according to the degree of matching with the intent of the text. \n Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n.'
        # q = f'Now gives 77 descriptions of specific intents and 1 specific text,specific text:{text}\n Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n.'
        # query_list.append(q)

        if result['category'][index] in candidate and tmpnum[result['category'][index]] < 40:
            query_list.append(q)
            text_new.append(text)
            label_new.append(result['category'][index])
            tmpnum[result['category'][index]] += 1

    newresult = {'query':query_list, 'text':text_new, 'category':label_new}
    result = pd.DataFrame(newresult)

    # result['query'] = query_list

    train_dataset = Dataset.from_pandas(result)

    def encode(item):
        temp_data = {}

        text = "### Query:" + item['query'] + '\n### Answer:'

        # 随机测试排序列表
        text += '1.' + item['category'] + '\n'
        randomlist = random.sample(labellist, 4)
        for i in range(2,6):
            text += str(i) + '.' + randomlist[i - 2] + '\n'


        # for i in range(1, 6):
        #     if pd.isna(item['top' + str(i)]): continue
        #     text += str(i) + '.' + item['top' + str(i)] + '\n'
        temp_data['text_answer'] = text
        return temp_data

    train_dataset = train_dataset.map(encode, batched=False, num_proc=16)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.lr,
        weight_decay = 5e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.95,
        num_train_epochs = args.num_train_epochs,
        # logging_dir=os.path.join(args.output_dir,'/logs'),
        fp16 = True,
        logging_strategy = "steps",
        logging_steps = 10,
        save_strategy = "epoch",
        report_to = 'none',
        deepspeed = args.deepspeed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = train_dataset,
        dataset_text_field = "text_answer",
        # max_seq_length = 2048,
        # peft_config=peft_config,
    )

    trainer.train()

if __name__ == '__main__':
    main()

