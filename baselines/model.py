import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import copy


class BackboneWithClsHead(nn.Module):
    """
    model_type:
    1. NT
    2. ESM
    3. NT+ESM (concat)
    4. NT+NT (concat)
    5. ESM+ESM (concat)
    """

    def __init__(self,
                 model_type: str,
                 nt_model: str = None,
                 esm_model: str = None,
                 num_labels: int = 2,
                 multi_label: bool = False
                 ):
        super(BackboneWithClsHead, self).__init__()
        self.model_type = model_type
        self.multi_label = multi_label

        if model_type == "NT":
            self.backbone = self._get_nt_model(nt_model)
            dim = self.backbone.config.hidden_size
        elif model_type == "ESM":
            self.backbone = self._get_esm_model(esm_model)
            dim = self.backbone.config.hidden_size
        elif model_type == "NT+ESM":
            self.nt = self._get_nt_model(nt_model)
            self.esm, self.tok_esm = self._get_esm_model_and_token
            dim = self.nt.config.hidden_size + self.esm.config.hidden_size
        elif model_type == "NT+NT":
            self.nt1 = self._get_nt_model(nt_model)
            self.nt2 = copy.deepcopy(self._get_nt_model(nt_model))
            dim = 2 * self.nt1.config.hidden_size
        elif model_type == "ESM+ESM":
            self.esm1 = self._get_esm_model(esm_model)
            self.esm2 = copy.deepcopy(self._get_esm_model(esm_model))
            dim = 2 * self.esm1.config.hidden_size
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        self.head = nn.Linear(dim, num_labels)

    def _get_esm_model(self, model_name_or_path):
        esm_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cuda:0",
        )
        return esm_model

    def _get_nt_model(self, model_name_or_path):
        nt_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cuda:0",
        )
        return nt_model


    def _cls(self, model, x):
        out = model(**x, output_hidden_states=True)
        last_hidden_state = out.hidden_states[-1]
        return last_hidden_state[:, 0]


    def forward(self, x1, x2=None, mask1=None, mask2=None, labels=None):
        if self.model_type == "NT":
            x = {
                'input_ids': x1,
                'attention_mask': mask1
            }
            h = self._cls(self.backbone, x)
        elif self.model_type == "ESM":
            x = {
                'input_ids': x1,
                'attention_mask': mask1
            }
            h = self._cls(self.backbone, x)
        elif self.model_type == "NT+ESM":
            # x1 = self.tok_nt(batch['seq'], return_tensors='pt', padding=True, truncation=True)
            # x1 = {k: v.to(self.nt.device) for k, v in x1.items()}
            # h1 = self._cls(self.nt, x1)

            # x2 = self.tok_esm(batch['seq'], return_tensors='pt', padding=True, truncation=True)
            # x2 = {k: v.to(self.esm.device) for k, v in x2.items()}
            # h2 = self._cls(self.esm, x2)

            # h = torch.cat([h1, h2], dim=-1)
            pass
        elif self.model_type == "NT+NT":
            x1 = {
                'input_ids': x1,
                'attention_mask': mask1
            }
            x2 = {
                'input_ids': x2,
                'attention_mask': mask2
            }
            h1 = self._cls(self.nt1, x1)
            h2 = self._cls(self.nt2, x2)
            h = torch.cat([h1, h2], dim=-1)
        elif self.model_type == "ESM+ESM":
            x1 = {
                'input_ids': x1,
                'attention_mask': mask1
            }
            x2 = {
                'input_ids': x2,
                'attention_mask': mask2
            }
            h1 = self._cls(self.esm1, x1)
            h2 = self._cls(self.esm2, x2)
            h = torch.cat([h1, h2], dim=-1)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        logits = self.head(h)

        loss = None
        if labels is not None:
            if self.multi_label:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )    

    def freze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
