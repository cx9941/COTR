import torch
from transformers import BertModel

class BERTForTextMatching(torch.nn.Module):
    def __init__(self, model_name):
        super(BERTForTextMatching, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, 128)  # Embedding size to 128

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)  # Return the embedding