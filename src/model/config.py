import copy

from transformers import AutoConfig, BertConfig

def get_qwen_bert_config(text_model_path, bio_model_path):
    # 加载子模型配置
    text_config = AutoConfig.from_pretrained(text_model_path, trust_remote_code=True)
    bio_config = BertConfig.from_pretrained(bio_model_path, trust_remote_code=True)

    # 构造复合模型配置
    model_config = copy.deepcopy(text_config)  # 避免共享引用
    model_config.text_config = text_config
    model_config.multimodal_model_config = bio_config

    return model_config