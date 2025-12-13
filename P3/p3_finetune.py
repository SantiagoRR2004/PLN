from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import EvalPrediction
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
    model: BertForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    )
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        # Freeze all BERT layers
        if name.startswith("bert."):
            param.requires_grad = False

        # Print the trainable parameters
        else:
            trainable_params += param.numel()
            print(f"{name}: {param.numel()}")

    print(f"Trainable parameters: {trainable_params}/{total_params}")
    # 2: Tokeniza el dataset
    print(dataset.column_names)
    tokenizedDataset = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=256
        ),
        batched=True,
        remove_columns=[
            "text"
        ],  # Eliminar la columana de texto original, ya no es necesaria. Usará los tokens
    )
    print(tokenizedDataset.column_names)
    # 3: Usa la clase TrainingArguments para definir las opciones de entrenamiento
    trainingArgs = TrainingArguments(
        output_dir=os.path.join(currentDirectory, "checkpoints"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",  # Por defecto es steps, con step = 500 ("no", "epoch", "steps")
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=0.00005,
        weight_decay=0.01,
    )

    def obtainAccuracy(data: EvalPrediction) -> dict:
        """
        4: Implementa una función para obtener la precisión durante
        la evaluación, a partir del par (logits, labels)

        compute_metrics : Callable[[EvalPrediction], Dict], optional

        Args:
            - data (EvalPrediction): Un objeto EvalPrediction contiene:
                - logits (torch.Tensor): lista de tuplas con las puntuaciones para cada clase
                - labels (torch.Tensor): lista con las clases para cada ejemplo de evaluación

        Returns:
            - dict: precisión del modelo en la evaluación
        """
        logits = data.predictions
        labels = data.label_ids
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
    print("Evaluating the model...")
    evalResults = trainer.evaluate()
    print(evalResults)
