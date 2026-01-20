# multimodal_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel, AutoConfig

class MultimodalConfig(PretrainedConfig):
    model_type = "multimodal_custom"

    def __init__(self, base_model_name="roberta-base", num_meta_features=3, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_meta_features = num_meta_features
        self.num_labels = num_labels

class MultimodalClassifier(PreTrainedModel):
    config_class = MultimodalConfig

    def __init__(self, config):
        super().__init__(config)
        self.text_model = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.text_model.config.hidden_size

        self.meta_fc = nn.Sequential(
            nn.Linear(config.num_meta_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_size + 128, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        meta_features=None,
        labels=None,
        **kwargs
    ):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]          # [batch, hidden]
        meta_emb = self.meta_fc(meta_features)                 # [batch, 64]
        fused = torch.cat([cls_emb, meta_emb], dim=1)          # [batch, hidden+64]
        logits = self.classifier(fused)                        # [batch, num_labels]

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
