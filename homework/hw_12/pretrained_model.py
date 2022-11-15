from transformers import (
    pipeline,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from huggingface_hub import notebook_login
from datasets import load_metric, load_dataset
import numpy as np



notebook_login()

phrases = [
    "I love you",
    "I hate you",
    'A math teacher cannot solve math problems',
    'People who disrespect women are born from them',
    'The pilot has a height phobia',
    'A cop gets robbed'
]

# def playground():
#     models = [
#         'finiteautomata/bertweet-base-sentiment-analysis',
#         'nlptown/bert-base-multilingual-uncased-sentiment',
#         'siebert/sentiment-roberta-large-english',
#         'sbcBI/sentiment_analysis_model',
#         'j-hartmann/sentiment-roberta-large-english-3-classes',
#         'sbcBI/sentiment_analysis',
#         'ahmedrachid/FinancialBERT-Sentiment-Analysis',
#     ]
#
#     for model in models:
#         specific_model = pipeline(model=model)
#         print(f"{model=} -> {specific_model(phrases)}")
#
# playground()

class DataProvider:
    def __init__(self, samples=3000):
        self.samples = samples
        imdb = load_dataset("imdb")
        small_train_dataset = imdb["train"].shuffle(seed=13).select([i for i in list(range(self.samples))])
        small_test_dataset = imdb["test"].shuffle(seed=13).select([i for i in list(range(self.samples))])

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tokenized_train = small_train_dataset.map(self.preprocess_function, batched=True)
        self.tokenized_test = small_test_dataset.map(self.preprocess_function, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)


class ModelManager:
    def __init__(self, data_provider, epochs=10, lr=0.00001):
        self.data_provider = data_provider
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        self.repo_name = f"finetuned-sentiment-model-{data_provider.samples}-samples"
        self.training_args = TrainingArguments(
            output_dir=self.repo_name,
            learning_rate=lr,
            per_device_train_batch_size=5,  # has impact on cuda/gpu
            per_device_eval_batch_size=5,  # has impact on cuda/gpu
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            push_to_hub=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=data_provider.tokenized_train,
            eval_dataset=data_provider.tokenized_test,
            tokenizer=data_provider.tokenizer,
            data_collator=data_provider.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def compute_metrics(self, eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f_one = load_metric("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f_one = load_f_one.compute(predictions=predictions, references=labels)["f1"]

        return {"accuracy": accuracy, "f1": f_one}

    def train_and_push(self, evaluate=True, push=True):
        self.trainer.train()
        if evaluate:
            evaluate_result = self.trainer.evaluate()
            print(f"Evaluate result is {evaluate_result}")

        if push:
            result = self.trainer.push_to_hub()
            print(result)


data_provider = DataProvider(5000)
manager = ModelManager(data_provider)

manager.train_and_push()

specific_model = pipeline(model=f'OleksiiF/{manager.repo_name}')
print(specific_model(phrases))
