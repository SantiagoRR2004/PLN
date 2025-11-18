from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from datasets import load_dataset
import torch

# Carga el dataset
dataset = load_dataset("imdb")

# TODO: Keep only train and test splits (remove unsupervised)


# 1: Carga el tokenizador y el modelo a refinar
model_name = "prajjwal1/bert-tiny"  # or "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2: Tokeniza el dataset
tokenizedDataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# TODO 3: Usa la clase TrainingArguments para definir las opciones de entrenamiento
trainingArgs = TrainingArguments()


def obtainAccuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    4: Implementa una función para obtener la precisión durante
    la evaluación, a partir del par (logits, labels)

    Args:
        - logits (torch.Tensor): lista de tuplas con las puntuaciones para cada clase
        - labels (torch.Tensor): lista con las clases para cada ejemplo de evaluación

    Returns:
        - float: precisión del modelo en la evaluación
    """
    return (logits.argmax(axis=1) == labels).mean().item()


# TODO 5: Usa la clase Trainer para definir un "entrenador" del modelo

# TODO 6: Entrena/refina el modelo, evalúa su rendimiento y muestra su precisión
