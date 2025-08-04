from dataclasses import dataclass
from typing import Optional
from transformers import AutoConfig, EsmConfig

@dataclass
class OmicsModalConfig:
    """Configuration class for multimodal model combining Qwen and DNA-BERT."""
    
    # Text model configuration
    text_config: Optional[dict] = None
    
    # DNA/RNA model configuration
    dna_rna_config: Optional[dict] = None

    # Protein model configuration
    protein_config: Optional[dict] = None
    
    text_max_length: int = 2048
    dna_rna_project_token_num: int = 64
    dna_rna_max_length: int = 512
    protein_project_token_num: int = 64
    protein_max_length: int = 512

    
    # Training configuration
    gradient_checkpointing: bool = False
    use_cache: bool = False
    
    def __post_init__(self):
        """Convert dictionary configs to proper config objects if needed."""
        if isinstance(self.text_config, dict):
            self.text_config = AutoConfig.from_pretrained(self.text_config['model_name'], **self.text_config.get('config', {}))
        if isinstance(self.dna_rna_config, dict):
            self.dna_rna_config = EsmConfig.from_pretrained(self.dna_rna_config['model_name'], **self.dna_rna_config.get('config', {}))
        if isinstance(self.protein_config, dict):
            self.protein_config = EsmConfig.from_pretrained(self.protein_config['model_name'], **self.protein_config.get('config', {}))


def get_omics_one_config(text_model_path, dna_rna_model_path, protein_model_path):
    """
    Create a configuration for the multimodal model.
    
    Args:
        text_model_path: Path to the Qwen model or configuration
        dna_rna_model_path: Path to the DNA/RNA model or configuration
        protein_model_path: Path to the protein model or configuration
        
    Returns:
        OmicsModalConfig: Config object containing all necessary configurations.
    """
    # Load base configurations
    text_config = AutoConfig.from_pretrained(text_model_path, trust_remote_code=True)
    dna_rna_config = AutoConfig.from_pretrained(dna_rna_model_path, trust_remote_code=True)
    protein_config = AutoConfig.from_pretrained(protein_model_path, trust_remote_code=True)

    # Create multimodal configuration
    config = OmicsModalConfig(
        text_config=text_config,
        dna_rna_config=dna_rna_config,
        protein_config=protein_config
    )
    
    # Add model-specific configurations
    config.text_config.use_cache = config.use_cache
    config.dna_rna_config.use_cache = config.use_cache
    config.protein_config.use_cache = config.use_cache

    config.text_config.gradient_checkpointing = config.gradient_checkpointing
    config.dna_rna_config.gradient_checkpointing = config.gradient_checkpointing
    config.protein_config.gradient_checkpointing = config.gradient_checkpointing
    
    return config
