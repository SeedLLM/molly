import copy
from dataclasses import dataclass
from typing import Optional
from transformers import AutoConfig, BertConfig

@dataclass
class MultimodalConfig:
    """Configuration class for multimodal model combining Qwen and DNA-BERT."""
    
    # Text model configuration
    text_config: Optional[dict] = None
    
    # DNA-BERT configuration
    bio_config: Optional[dict] = None
    
    # Multimodal configuration
    project_token_num: int = 64  # Number of tokens to project DNA sequences to
    dna_max_length: int = 512    # Maximum length of DNA sequences
    text_max_length: int = 2048  # Maximum length of text sequences
    
    # Special tokens
    dna_start_token: str = "<|dna_start|>"
    dna_end_token: str = "<|dna_end|>"
    dna_pad_token: str = "<|dna_pad|>"
    
    # Training configuration
    gradient_checkpointing: bool = False
    use_cache: bool = True
    
    def __post_init__(self):
        """Convert dictionary configs to proper config objects if needed."""
        if isinstance(self.text_config, dict):
            self.text_config = AutoConfig.from_pretrained(self.text_config['model_name'], **self.text_config.get('config', {}))
        if isinstance(self.bio_config, dict):
            self.bio_config = BertConfig.from_pretrained(self.bio_config['model_name'], **self.bio_config.get('config', {}))

def get_qwen_bert_config(text_model_path, bio_model_path):
    """
    Create a configuration for the multimodal model.
    
    Args:
        text_model_path: Path to the Qwen model or configuration
        bio_model_path: Path to the DNA-BERT model or configuration
        
    Returns:
        MultimodalConfig: Configuration for the multimodal model
    """
    # Load base configurations
    text_config = AutoConfig.from_pretrained(text_model_path, trust_remote_code=True)
    bio_config = BertConfig.from_pretrained(bio_model_path, trust_remote_code=True)

    # Create multimodal configuration
    config = MultimodalConfig(
        text_config=text_config,
        bio_config=bio_config
    )
    
    # Add model-specific configurations
    config.text_config.use_cache = config.use_cache
    # Set use_cache to False to avoid DynamicCache issues during distributed evaluation
    config.text_config.use_cache = False 
    if config.gradient_checkpointing:
        config.text_config.gradient_checkpointing = True
        config.bio_config.gradient_checkpointing = True
    
    return config