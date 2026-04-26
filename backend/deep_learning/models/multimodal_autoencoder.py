import torch
import torch.nn as nn
from transformers import DistilBertModel

class TabularBranch(nn.Module):
    def __init__(self, tabular_input_dim, hidden_dim=64):
        super(TabularBranch, self).__init__()
        # Expands the small number of numeric limits (approx 8-10 dimensions)
        # into a rich dense space before fusion
        self.net = nn.Sequential(
            nn.Linear(tabular_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class TextBranch(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", freeze_encoder=True):
        super(TextBranch, self).__init__()
        # Load the base transformer model (Encoder only)
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze weights to save massive amounts of RAM and compute
        # since we just want it as a fixed semantic feature extractor initially
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # DistilBERT outputs 768 dimensional vectors
        self.output_dim = self.bert.config.dim 
        
    def forward(self, input_ids, attention_mask):
        # DistilBert returns a tuple, first element is last hidden state
        # Shape: (batch_size, sequence_length, hidden_dim)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        # Get the CLS token representation (first token)
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding

class MultimodalAutoencoder(nn.Module):
    def __init__(self, tabular_input_dim, tabular_hidden_dim=64, bottleneck_dim=16, freeze_text=True):
        super(MultimodalAutoencoder, self).__init__()
        
        # 1. Initiating the Branches
        self.tabular_branch = TabularBranch(tabular_input_dim, tabular_hidden_dim)
        self.text_branch = TextBranch(freeze_encoder=freeze_text)
        
        # Calculate combined dimension: e.g. 768 (DistilBERT) + 64 (Tabular) = 832
        fusion_dim = self.text_branch.output_dim + tabular_hidden_dim
        
        # 2. Encoder Fusion Layer (Compressing down to Bottleneck)
        self.encoder_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, bottleneck_dim)  # This is the Student Embedding
        )
        
        # 3. Task-Specific Decoder (Reconstructing Tabular data)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, tabular_input_dim) # Predict standardized numeric output
        )
        
        # 4. Text Decoder (Reconstructing 768-D DistilBERT Embedding)
        self.text_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.text_branch.output_dim) # 768
        )

    def encode(self, tabular_x, input_ids=None, attention_mask=None, precomputed_text=None):
        """
        Runs both branches and outputs the fused bottleneck embedding.
        Used natively during inference/production.
        """
        tab_features = self.tabular_branch(tabular_x)
        
        if precomputed_text is not None:
            text_features = precomputed_text
        else:
            text_features = self.text_branch(input_ids, attention_mask)
        
        # Weight text slightly higher prior to fusion (User Request)
        weighted_text = text_features * 1.5
        fused = torch.cat((tab_features, weighted_text), dim=1)
        bottleneck = self.encoder_fusion(fused)
        return bottleneck


    def forward(self, tabular_x, input_ids=None, attention_mask=None, precomputed_text=None):
        """
        Full Forward pass used during Training. 
        Compresses modalities then attempts to rebuild the numerical data and text embeddings.
        """
        # 1. Generate Embeddings (Encoder)
        bottleneck = self.encode(tabular_x, input_ids=input_ids, attention_mask=attention_mask, precomputed_text=precomputed_text)
        
        # 2. Predict original features (Decoders)
        reconstructed_tabular = self.decoder(bottleneck)
        reconstructed_text = self.text_decoder(bottleneck)
        return reconstructed_tabular, reconstructed_text, bottleneck
