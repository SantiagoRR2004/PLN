from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, DatasetDict
import utils
import os

# Obtain the current directory
currentDirectory = os.path.dirname(os.path.abspath(__file__))


def obtainDataset() -> DatasetDict:
    """
    Loads the IMDb dataset and removes the unsupervised split.

    Args:
        - None

    Returns:
        - DatasetDict: The IMDb dataset with only train and test splits.
    """
    # Load the IMDb dataset
    dataset: DatasetDict = load_dataset("imdb")

    # Keep only train and test splits (remove unsupervised)
    del dataset["unsupervised"]

    return dataset


if __name__ == "__main__":
    # Check if GPU is available and use it
    utils.canUseGPU()

    # Load the dataset
    dataset = obtainDataset()

    # 1: Carga el tokenizador y el modelo a refinar
    model_name = "prajjwal1/bert-tiny"  # or "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    for param in model.bert.parameters():
        param.requires_grad = False

    # 2: Tokeniza el dataset
    tokenizedDataset = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=256
        ),
        batched=True,
    )

    # 3: Usa la clase TrainingArguments para definir las opciones de entrenamiento
    trainingArgs = TrainingArguments(
        output_dir=os.path.join(currentDirectory, "checkpoints"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
    )

    def obtainAccuracy(data: tuple) -> dict:
        """
        4: Implementa una función para obtener la precisión durante
        la evaluación, a partir del par (logits, labels)

        Args:
            - data (tuple): A tuple containing:
                - logits (torch.Tensor): lista de tuplas con las puntuaciones para cada clase
                - labels (torch.Tensor): lista con las clases para cada ejemplo de evaluación

        Returns:
            - dict: precisión del modelo en la evaluación
        """
        logits, labels = data[0], data[1]
        acc = (logits.argmax(axis=1) == labels).mean().item()
        return {"accuracy": acc}

    # 5: Usa la clase Trainer para definir un "entrenador" del modelo
    trainer = Trainer(
        model=model,
        args=trainingArgs,
        train_dataset=tokenizedDataset["train"],
        eval_dataset=tokenizedDataset["test"],
        compute_metrics=obtainAccuracy,
    )

    # 6: Entrena/refina el modelo, evalúa su rendimiento y muestra su precisión
    trainer.train()
    evalResults = trainer.evaluate()
    print(evalResults)
