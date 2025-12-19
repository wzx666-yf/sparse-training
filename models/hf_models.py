import torch
from torch import nn

try:
    from transformers import GPT2LMHeadModel, BertForSequenceClassification
except Exception:
    GPT2LMHeadModel = None
    BertForSequenceClassification = None

class GPT2SmallLM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        assert GPT2LMHeadModel is not None, 'transformers not installed'
        model_id = 'gpt2'
        self.m = GPT2LMHeadModel.from_pretrained(model_id) if pretrained else GPT2LMHeadModel.from_pretrained(model_id)
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, labels=labels if labels is not None else input_ids)

class GPT2MediumLM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        assert GPT2LMHeadModel is not None, 'transformers not installed'
        model_id = 'gpt2-medium'
        self.m = GPT2LMHeadModel.from_pretrained(model_id) if pretrained else GPT2LMHeadModel.from_pretrained(model_id)
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, labels=labels if labels is not None else input_ids)

class BertLargeSST2(nn.Module):
    def __init__(self, num_labels=2, pretrained=True):
        super().__init__()
        assert BertForSequenceClassification is not None, 'transformers not installed'
        self.m = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=num_labels)
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
