import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import copy


class EvoWrapper(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        from evo2 import Evo2
        self.evo = Evo2(model_name_or_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evo.model.to(device)

        self.blocks = self.evo.model.blocks

    def forward(self, *args, **kwargs):
        return self.evo(*args, **kwargs)

    @property
    def config(self):
        class C:
            hidden_size = 1920
        return C()

    def last_layer_name(self):
        return f"blocks.23.mlp.l3"

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
                 multi_label: bool = False,
                 multi_answer: bool = False
                 ):
        super(BackboneWithClsHead, self).__init__()
        self.model_type = model_type
        self.multi_label = multi_label
        self.multi_answer = multi_answer

        if model_type == "NT":
            self.backbone = self._get_nt_model(nt_model)
            dim = self.backbone.config.hidden_size
        elif model_type == "ESM":
            self.backbone = self._get_esm_model(esm_model)
            dim = self.backbone.config.hidden_size
        elif model_type == "NT+ESM":
            self.nt = self._get_nt_model(nt_model)
            self.esm = self._get_esm_model(esm_model)
            dim = self.nt.config.hidden_size + self.esm.config.hidden_size
        elif model_type == "NT+NT":
            self.nt1 = self._get_nt_model(nt_model)
            self.nt2 = copy.deepcopy(self._get_nt_model(nt_model))
            dim = 2 * self.nt1.config.hidden_size
        elif model_type == "ESM+ESM":
            self.esm1 = self._get_esm_model(esm_model)
            self.esm2 = copy.deepcopy(self._get_esm_model(esm_model))
            dim = 2 * self.esm1.config.hidden_size
        elif model_type == "EVO":
            self.backbone = self._get_evo_model(nt_model)
            dim = 1920
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

    def _get_evo_model(self, model_name_or_path):
        return EvoWrapper(model_name_or_path)


    def _cls(self, model, x):
        if self.model_type == "EVO":
            input_ids = x['input_ids']
            layer_name = model.last_layer_name()
            outputs, embeddings = model(
                input_ids,
                return_embeddings=True,
                layer_names=[layer_name]
            )
            h = embeddings[layer_name]
            mask = x['attention_mask'].unsqueeze(-1)
            h = (h * mask).sum(dim=1) / mask.sum(dim=1)
            return h
        else:
            out = model(**x, output_hidden_states=True)
            last_hidden_state = out.hidden_states[-1]
            return last_hidden_state[:, 0]       # [CLS] token


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
            x1 = {
                'input_ids': x1,
                'attention_mask': mask1
            }
            x2 = {
                'input_ids': x2,
                'attention_mask': mask2
            }
            h1 = self._cls(self.nt, x1)
            h2 = self._cls(self.esm, x2)
            h = torch.cat([h1, h2], dim=-1)
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
        elif self.model_type == "EVO":
            x = {
                'input_ids': x1,
                'attention_mask': mask1
            }
            h = self._cls(self.backbone, x)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        # logits = self.head(h)
        # print(f"h shape: {h.shape}")
        # print(h)
        # print(self.head.weight.dtype)
        logits = self.head(h.to(self.head.weight.dtype))

        # print(f"labels shape: {labels.shape}")
        # print(labels)

        loss = None
        if labels is not None:
            if self.multi_answer:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                # print("logits", logits)
                # print("labels", labels)
                loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )    

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
