import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm
tqdm.pandas()

task_keywords = {
    "fr": ["Responsabilité", "Responsable", "responsabilités", "Activités", 'à faire concrètement dans', "attentes", "devoir", 'SOMMAIRE DES FONCTIONS','aurez principalement à', 'le titulaire', 'il aura à', 'fonctions', 'responsabilités', 'Plus précisément', 'tâches','Rôle'],
    "eu": ['Responsibilities','responsible','Responsibilties','Activities','expectations','Duties', 'role'],
    "en": ['Responsibilities','responsible','Responsibilties','Activities'],
    "jp": ['勤務内容', '業務内容','仕事内容','作業内容','主なサービス内容'],
}

task_symbolwords = {
    "en": [':'],
    "eu": [':'],
    "fr": [':'],
    "jp": [':', '：', '\n'],
}

task_must_words = {
    "fr": [],
    "eu": [],
    "en": [],
    "jp": ['以下業務'],
}

symbol_line = {
    "fr": [],
    "jp": ["*", "·", '★', '■', '◆', '・', 'ー', '◼︎', '-'],
    "en": ["*", "·", '★', '■', '◆', '・', 'ー', '◼︎', '-'],
    "eu": [],
}

stopwords = {
    "fr": [],
    "jp": [':', '：', '【', '\n\s*\n', '必須スキル'],
    "en": [':', '\n', 'Requirements', 'Qualifications'],
    "eu": [],
}

splitwords = {
    "fr": [],
    # "jp": ['。', '\n'],
    "jp": ["*", '★', '■', '◆', 'ー ',' ー', '◼︎',  '。', '\n', '◇'],
    "en": ['.', ';', '*', '—', '·', '–', '\n', '•', '●', '-'],
    "eu": [],
}
special_word_match = {
    "fr": r'([^a-zA-ZàâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+)\s*',
    "jp": r'([^\u3040-\u30FF\u4E00-\u9FFFa-zA-Z]|[・ー])+\s*',
    "en": r'^([\W\d]+)\s*',
    "eu": r'^([\W\d]+)\s*',
}
s_word_list = {
    "fr": [':', '\?', '：', '？'],
    "jp": [':', '\?', '：', '？', '】', '【', '★', '■', '＞', '＜', '◆', '・', 'ー', '◼︎'],
    "en": [':', '\?', '：', '？'],
    "eu": [':', '\?', '：', '？']
}

class Task_Extraction():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.task_keywords = task_keywords[dataset_name]
        self.symbol_line = symbol_line[dataset_name]
        self.splitwords = splitwords[dataset_name]
        self.stopwords = stopwords[dataset_name]
        self.task_must_words = task_must_words[dataset_name]
        self.special_word_match = special_word_match[dataset_name]
        self.task_symbolwords = task_symbolwords[dataset_name]
        self.s_word_list = s_word_list[dataset_name]

        self.action_verbs = ["Prepare", "Organize", "Design", "Lead", "Complete", "Participate", "Managing", "Track", "Plan", "Communicate", "Manage"] + ["Manage", "Develop", "Plan", "Coordinate", "Implement", "Prepare", "Organize", "Oversee", "Lead", "Execute", "Support", "Monitor", "Receive", "Maintain"]
        self.action_verbs = list(set(self.action_verbs))
        self.keywords = ["develop", "prepare", "coordinate", "oversee", "lead", "support","is commited to"]

    def start_symbol(self, line):
        task_start = '|'.join(self.task_keywords)
        if len(re.findall(rf'{task_start}', line, re.IGNORECASE)) > 0 and any(s_word in line for s_word in self.s_word_list):
            return True
        if len(self.task_must_words) > 0:
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
        

    def extract_tasks_from_description_en(self, text):
        keyword_pattern = r"|".join(self.task_keywords)
        split_pattern = "".join([re.escape(word) for word in self.splitwords])
        stop_pattern = r"|".join(self.stopwords)
        task_pattern = "".join(self.task_symbolwords)
        # pattern = rf"(?:{keyword_pattern}).{{0,20}}{task_pattern}(.*?[{split_pattern}]).*?(?={stop_pattern})"
        pattern = rf"(?:{keyword_pattern}).{{0,20}}[{task_pattern}].{{0,20}}[{split_pattern}]\s*(.*?)(?={stop_pattern})"

        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) > 0:
            ans = [j for i in matches for j in re.split(r'\.|;|\*|—|\n|·|–', i)]
        else:
            ans = []
        return ans

    def extract_tasks_from_description_jp(self, text):
        # text = re.sub(r'\n\s+', '\n', text)
        keyword_pattern = r"|".join(self.task_keywords)
        split_pattern = "".join([re.escape(word) for word in self.splitwords])
        stop_pattern = r"|".join(self.stopwords)
        task_pattern = "".join(self.task_symbolwords)
        # pattern = rf"(?:{keyword_pattern}).{{0,1}}[{task_pattern}].{{0,9}}(.*?[{split_pattern}]).*?(?={stop_pattern})"
        pattern = rf"(?:{keyword_pattern}).{{0,1}}[{task_pattern}].{{0,9}}[{split_pattern}]\s*(.*?)(?={stop_pattern})"
        
        matches = re.findall(pattern, text, re.IGNORECASE| re.DOTALL)

        split_pattern = "|".join([re.escape(word) for word in self.splitwords])
        if len(matches) > 0:
            ans = [j for i in matches for j in re.split(split_pattern, i)]
        else:
            ans = []
        return ans

    def extract_tasks_from_description_eu(self, text):
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
            match = re.match(self.special_word_match, line[:1])

            # if any(s_word in line for s_word in self.s_word_list):
            #     break
            
            if match:
                if special_symbol is None:
                    # Set the first detected special symbol as the pattern to follow
                    special_symbol = re.escape(match.group(1))  # Escape special characters for regex usage
                    if re.findall(r'\d', special_symbol):
                        special_symbol = '\d'

                # Only collect lines that match the detected special symbol
                if len(special_symbol)<=4 and re.match(f'^{special_symbol}\s*', line):
                    # Remove the special symbol and add the task description
                    # task = re.sub(f'^{special_symbol}\s*', '', line)
                    task = line.split(special_symbol)
                    task = self.filter_task(task)
                    task_list += task
                else:
                    break  # Stop if we encounter a line without the special symbol
            else:
                # Stop if the line doesn't match the special symbol pattern and we already found tasks
                break

        new_task_list = self.extract_tasks_from_description_eu('\n'.join(lines[responsibilities_start+1:]))

        task_list = new_task_list if len(new_task_list) > len(task_list) else task_list

        if len(task_list) == 1:
            task_list = task_list[0].split(special_symbol)

        return task_list

    def extract_tasks_from_description_eu2(self, text):
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
        
        task_list = []
        for i in range(responsibilities_start + 1, len(lines)):
            line = lines[i].strip()
            if self.end_symbol(line):
                break
            else:
                task_list.append(line)

        return task_list

    def filter_task(self, task_list):
        ans = []
        for line in task_list:
            if line.startswith('.'):
                line = line[1:]
            line = line.strip(' ')
            if ('jp' in self.dataset_name and len(line) <= 2) or ('jp' not in self.dataset_name and len(line.split(' ')) <= 4):
                continue
            ans.append(line)
        return ans
    
    def extract_tasks_from_description(self, text):
        if self.dataset_name in ['en']:
            task_list = self.extract_tasks_from_description_en(text)
        elif self.dataset_name in ['jp']:
            task_list = self.extract_tasks_from_description_jp(text)
        elif self.dataset_name in ['eu']:
            task_list = self.extract_tasks_from_description_eu(text)
            if len(task_list) == 0:
                task_list = self.extract_tasks_from_description_eu2(text)
        task_list = self.filter_task(task_list)
        return task_list

    
if __name__ == '__main__':
    dataset_name = 'jp'
    task = Task_Extraction(dataset_name)
    content = open(f'data/{dataset_name}/temp.txt').read()
    task_list = task.extract_tasks_from_description(content)
    print('\n'.join(task_list))