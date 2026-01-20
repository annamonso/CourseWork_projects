# data_collator_mm.py
from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class DataCollatorWithMeta:
    tokenizer: any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # extraer metadatos y etiquetas
        meta = torch.stack([f["meta_features"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        # preparar entradas de texto (input_ids, attention_mask)
        batch_text = self.tokenizer.pad(
            [{k: v for k, v in feat.items() if k in ["input_ids", "attention_mask"]} for feat in features],
            padding=True,
            return_tensors="pt"
        )

        # añadir metadatos y etiquetas al batch
        batch_text["meta_features"] = meta
        batch_text["labels"] = labels
        return batch_text
