from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,DataCollatorForLanguageModeling,AutoModel
import argparse
import json
import torch
import pandas as pd

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',default="/home/data/qinchuan/TMIS/COTR/llms/Meta-Llama-3.1-8B-Instruct", type=str, help='')
    # parser.add_argument('--model_dir',default="public/baichuan-inc/Baichuan-13B-Base", type=str, help='')
    parser.add_argument('--output_dir', default='/individuaL/fangchuyu/icde/chatgpt_query/output/baichuan/',type=str,help='')
    parser.add_argument('--num_train_epochs',default=3,type=int,help='')
    parser.add_argument('--lr',default=3e-5,type=float,help='')
    parser.add_argument('--per_device_train_batch_size',default=1,type=int,help='')
    parser.add_argument('--gradient_accumulation_steps',default=8,type=int,help='')
    parser.add_argument('--local_rank',type=int,default=0,help='')
    # parser,add_argument('--lora_r',type=int,default=8,help=')
    parser.add_argument("--deepspeed",type=str,default='/individual/fangchuyu/Llmsft/ds_zero2.json',help="path to deepspeed config file.")
    return parser.parse_args()

def main():
    args = set_args()
    # idlabelList ={'clinc':'/individuaL/fangchuyu/task_classify/lotrdatasets/clinc/id2label.json','bank':'/individuaL/fangchuyu/task_classify/lotrdatasets/banking/id2label.json'}
    # idlabellist = {'task':'/individual/fangchuyu/task_classify/label.txt'}
    idlabellist = {'eu': '/home/data/qinchuan/TMIS/COTR/step3-bert_recall/data/eu/task.csv', 'en': '/home/data/qinchuan/TMIS/COTR/step3-bert_recall/data/en/task.csv', 'jp': '/home/data/qinchuan/TMIS/COTR/step3-bert_recall/data/jp/task.csv'}

    prompt_init_json = {}
    for task in idlabellist:
        idlabelpath = idlabellist[task]
        prompt_data_json ={}
        # with open(idlabelpath,'r')as f:
        #     label2id = json.load(f)
        labellist = pd.read_csv(idlabelpath, sep='\t')
        label2id = list(labellist['DWA Title'])


        # model_name_list =['baichuan','1lama','nanbeige']
        # model_path_list =['', '', '']
        model_name_list = ['llama', 'baichuan', 'nanbeige']
        model_path_list = ['/home/data/qinchuan/TMIS/COTR/llms/Meta-Llama-3.1-8B-Instruct', '/home/data/qinchuan/TMIS/paper_code/llm_models/baichuan-inc/Baichuan2-7B-Base', '/home/data/qinchuan/TMIS/COTR/llms/models--Nanbeige--Nanbeige2-8B-Chat/snapshots/b11c6c1b14b01bb42f861162e1049d08c4789ea2']
        for index, model_path in enumerate(model_path_list):
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            embed_layer = model.get_input_embeddings()
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            prefix_prompt = list(label2id)
            with torch.no_grad():
                prefix_encoded_inputs = tokenizer(prefix_prompt, max_length=1024, truncation=True, padding=True, return_tensors='pt')
                prefix_seq = prefix_encoded_inputs['input_ids']
                prefix_mask = prefix_encoded_inputs['attention_mask']

                ones_count = torch.sum(prefix_mask == 1).item()
                dim_size = prefix_mask.size(0)
                result = ones_count / dim_size
                print(result)
                w_pre = embed_layer(prefix_seq)
                prompt_init_embedding = torch.mean(w_pre, dim=1)
                prompt_data_json[model_name_list[index]] = prompt_init_embedding.numpy().tolist()

            prompt_init_json[task] = prompt_data_json
        with open('/home/data/qinchuan/TMIS/paper_code/data_obtain/prompt_init.json', 'w') as f:
            json.dump(prompt_init_json, f)

if __name__ == "__main__":
    main()
