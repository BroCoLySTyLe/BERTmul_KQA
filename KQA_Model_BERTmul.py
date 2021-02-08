import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from kobert.pytorch_kobert import get_pytorch_kobert_model
from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax
from transformers import BertModel
#model = BertModel.from_pretrained("bert-base-multilingual-cased")
class QuestionAnswering(nn.Module):
    def __init__(self):
        super(QuestionAnswering, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output,_ = self.bert(input_ids,attention_mask,token_type_ids)
#         sequence_output = output.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
