import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm
tqdm.pandas()

class Task_Extraction():
    def __init__(self, dataset_name):
        self.task_start_words = {
            "fr": ["Responsabilité", "Responsable", "responsabilités", "Activités", 'à faire concrètement dans', "attentes", "devoir", 'SOMMAIRE DES FONCTIONS','aurez principalement à', 'fonctions', 'responsabilités', 'Plus précisément', 'tâches','Rôle'],
            "eu": ['Responsibilities','responsible','Responsibilties','Activities','expectations','Duties', 'role'],
            "en": ['Responsibilities','responsible','Responsibilties','Activities','expectations','Duties', 'role'],
            "jp": ['職責', '勤務内容', '業務内容','ロール＃ロール＃']
        }
        self.dataset_name = dataset_name

        self.task_start_words = self.task_start_words[dataset_name]

        self.s_word_list = [':', '?', '：', '？']
        self.action_verbs = ["Prepare", "Organize", "Design", "Lead", "Complete", "Participate", "Managing", "Track", "Plan", "Communicate", "Manage"] + ["Manage", "Develop", "Plan", "Coordinate", "Implement", "Prepare", "Organize", "Oversee", "Lead", "Execute", "Support", "Monitor", "Receive", "Maintain"]
        self.action_verbs = list(set(self.action_verbs))
        self.keywords = ["develop", "prepare", "coordinate", "oversee", "lead", "support","is commited to"]


    # def extract_tasks_from_en_description(job_description):
    #     tasks = []

    #     if action_verbs:
    #         verb_pattern = r"|".join(action_verbs)
    #         delimiters = r'[\.!?\*\-]'
    #         task_pattern = re.compile(rf'\b(?:{verb_pattern})\b.{{4,}}?{delimiters}', re.DOTALL)
    #         tasks.extend(re.findall(task_pattern, job_description))
        
    #     if keywords:
    #         # 使用关键词列表进行匹配，捕获包含这些关键词的完整句子
    #         keyword_pattern = r"|".join(keywords)
    #         task_pattern = re.compile(rf'\b.{{2,}}?(?:{keyword_pattern})\b.{{4,}}?{delimiters}', re.DOTALL)
    #         tasks.extend(re.findall(task_pattern, job_description))
    #     tasks = list(set(tasks))
    #     return tasks

    def extract_tasks_from_eu_description(self, text):
        # Step 1: Split the text into lines
        lines = [i for i in text.splitlines() if i!='\n']

        # Step 2: Find the line with 'Responsibilities'
        responsibilities_start = None
        for i, line in enumerate(lines):
            task_start = '|'.join(self.task_start_words[:-1])
            if self.dataset_name != 'jp':
                if re.findall(rf'{task_start}', line, re.IGNORECASE) and any(s_word in line for s_word in self.s_word_list):
                    responsibilities_start = i
                    break
            else:
                if re.findall(rf'{task_start}', line):
                    responsibilities_start = i
                    break

        if responsibilities_start is None:
            return []

        # Step 3: Identify the first line with a special symbol after 'Responsibilities'
        task_list = []
        special_symbol = None
        for i in range(responsibilities_start + 1, len(lines)):
            line = lines[i].strip()
            
            # Match lines starting with special symbols (e.g., -, *, •, numbers followed by .)
            if self.dataset_name == 'jp':
                match = re.match(r'([^\u3040-\u30FF\u4E00-\u9FFFa-zA-Z]+)\s+', line)
            if self.dataset_name == 'fr':
                match = re.match(r'([^a-zA-Z àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+)\s+', line)
            else:
                match = re.match(r'^([\W\d]+)\s+', line)
                
            if match:
                if special_symbol is None:
                    # Set the first detected special symbol as the pattern to follow
                    special_symbol = re.escape(match.group(1))  # Escape special characters for regex usage
                    if len(special_symbol)<=4 and re.findall(r'\d', special_symbol):
                        special_symbol = '\d\\.'

                # Only collect lines that match the detected special symbol
                if len(special_symbol)<=4 and re.match(f'^{special_symbol}\s+', line):
                    # Remove the special symbol and add the task description
                    task = re.sub(f'^{special_symbol}\s+', '', line)
                    task_list.append(task)
                else:
                    break  # Stop if we encounter a line without the special symbol
            elif special_symbol:
                # Stop if the line doesn't match the special symbol pattern and we already found tasks
                break
        
        if len(task_list) == 0:
            # Step 2: Find the line with 'Responsibilities'
            responsibilities_start = None
            for i, line in enumerate(lines):
                task_start = '|'.join(self.task_start_words)
                if re.findall(rf'{task_start}', line, re.IGNORECASE) and any(s_word in line for s_word in self.s_word_list):
                    responsibilities_start = i
                    break

            if responsibilities_start is None:
                return []
            
            task_list = []
            for i in range(responsibilities_start + 1, len(lines)):
                line = lines[i].strip()
                if any(s_word in line for s_word in self.s_word_list) or len(line) < 6 if self.dataset_name in ['jp'] else len(line.split(' ')) < 4:
                    break
                else:
                    task_list.append(line)

        return task_list
    
if __name__ == '__main__':
    dataset_name = 'jp'
    task = Task_Extraction(dataset_name)
    content = open(f'data/{dataset_name}/temp.txt').read()
    task_list = task.extract_tasks_from_eu_description(content)
    print('\n'.join(task_list))