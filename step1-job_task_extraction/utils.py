import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm
tqdm.pandas()

class Task_Extraction():
    def __init__(self, dataset_name):
        self.task_start_words = {
            "fr": ["Responsabilité", "Responsable", "responsabilités", "Activités", 'à faire concrètement dans', "attentes", "devoir", 'SOMMAIRE DES FONCTIONS','aurez principalement à', 'le titulaire', 'il aura à', 'fonctions', 'responsabilités', 'Plus précisément', 'tâches','Rôle'],
            "eu": ['Responsibilities','responsible','Responsibilties','Activities','expectations','Duties', 'role'],
            "en": ['Responsibilities','responsible','Responsibilties','Activities','expectations','Duties', 'role'],
            "jp": ['職責', '勤務内容', '業務内容','仕事内容','作業内容','オープニング募集','担当業務の流れ','主なサービス内容','業務', 'ロール＃ロール＃'],
        }

        self.task_must_words = {
            "fr": [],
            "eu": [],
            "en": [],
            "jp": ['以下業務'],
        }


        self.special_word_match = {
            "fr": r'([^a-zA-ZàâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+)\s*',
            "jp": r'([^\u3040-\u30FF\u4E00-\u9FFFa-zA-Z]|[・ー])+\s*',
            "en": r'^([\W\d]+)\s*',
            "eu": r'^([\W\d]+)\s*',
        }
        self.s_word_list = {
            "fr": [':', '\?', '：', '？'],
            "jp": [':', '\?', '：', '？', '】', '【', '★'],
            "en": [':', '\?', '：', '？'],
            "eu": [':', '\?', '：', '？']
        }
        self.dataset_name = dataset_name
        self.task_start_words = self.task_start_words[dataset_name]
        self.special_word_match = self.special_word_match[dataset_name]
        self.s_word_list = self.s_word_list[dataset_name]

        self.action_verbs = ["Prepare", "Organize", "Design", "Lead", "Complete", "Participate", "Managing", "Track", "Plan", "Communicate", "Manage"] + ["Manage", "Develop", "Plan", "Coordinate", "Implement", "Prepare", "Organize", "Oversee", "Lead", "Execute", "Support", "Monitor", "Receive", "Maintain"]
        self.action_verbs = list(set(self.action_verbs))
        self.keywords = ["develop", "prepare", "coordinate", "oversee", "lead", "support","is commited to"]

    def start_symbol(self, line):
        task_start = '|'.join(self.task_start_words[:-1])
        if len(re.findall(rf'{task_start}', line, re.IGNORECASE)) > 0 and any(s_word in line for s_word in self.s_word_list):
            return True
        task_must = '|'.join(self.task_must_words)
        if len(re.findall(rf'{task_must}', line, re.IGNORECASE)) > 0:
            return True
        return False

    
    def end_symbol(self, line):
        if any(s_word in line for s_word in self.s_word_list):
            return True
        if 'jp' in self.dataset_name:
            return len(line) < 6
        else:
            return len(line.split(' ')) < 4


    def extract_tasks_from_description_step1(self, text):
        # Step 1: Split the text into lines
        lines = [i.strip(' ').replace('\ufeff', '')for i in text.splitlines()]
        lines = [i for i in lines if len(i)>0]

        # Step 2: Find the line with 'Responsibilities'
        responsibilities_start = None
        for i, line in enumerate(lines):
            if self.start_symbol(line):
                responsibilities_start = i
                break

        if responsibilities_start is None:
            return []

        # Step 3: Identify the first line with a special symbol after 'Responsibilities'
        task_list = []
        special_symbol = None
        for i in range(responsibilities_start + 1, len(lines)):
            line = lines[i].strip()
            # line = re.sub(self.sub_regex, '', line)
            # for sub in self.s_word_list:
            #     line = line.replace(sub, '')
            
            # Match lines starting with special symbols (e.g., -, *, •, numbers followed by .)
            match = re.match(self.special_word_match, line)

            if any(s_word in line for s_word in self.s_word_list):
                break
            
            if match:
                if special_symbol is None:
                    # Set the first detected special symbol as the pattern to follow
                    special_symbol = re.escape(match.group(1))  # Escape special characters for regex usage
                    if re.findall(r'\d', special_symbol):
                        special_symbol = '\d'

                # Only collect lines that match the detected special symbol
                if len(special_symbol)<=4 and re.match(f'^{special_symbol}\s*', line):
                    # Remove the special symbol and add the task description
                    task = re.sub(f'^{special_symbol}\s*', '', line)
                    task_list.append(task)
                else:
                    break  # Stop if we encounter a line without the special symbol
            elif special_symbol:
                # Stop if the line doesn't match the special symbol pattern and we already found tasks
                break

        if len(task_list) == 0:
            task_list = self.extract_tasks_from_description_step1('\n'.join(lines[responsibilities_start+1:]))

        return task_list
    
    def extract_tasks_from_description_step2(self, text):
        # Step 1: Split the text into lines
        lines = [i.strip(' ').replace('\ufeff', '')for i in text.splitlines()]
        lines = [i for i in lines if len(i)>0]

        # Step 2: Find the line with 'Responsibilities'
        responsibilities_start = None
        task_start = '|'.join(self.task_start_words)
        for i, line in enumerate(lines):
            if self.start_symbol(line):
                responsibilities_start = i
                break

        if responsibilities_start is None:
            return []
        
        task_list = []
        for i in range(responsibilities_start + 1, len(lines)):
            line = lines[i].strip()
            if self.end_symbol(line):
                break
            else:
                task_list.append(line)

        return task_list

    def filter_task(self, text):
        if text.startswith('.'):
            text = text[1:]
        return text.strip(' ')
    
    def extract_tasks_from_description(self, text):
        task_list = self.extract_tasks_from_description_step1(text)
        if len(task_list) == 0:
            task_list = self.extract_tasks_from_description_step2(text)
        task_list = [self.filter_task(i) for i in task_list]
        return task_list

    
if __name__ == '__main__':
    dataset_name = 'jp'
    task = Task_Extraction(dataset_name)
    content = open(f'data/{dataset_name}/temp.txt').read()
    task_list = task.extract_tasks_from_description(content)
    print('\n'.join(task_list))