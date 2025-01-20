import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from collections import Counter

try:
    # Stable Bias Professions veri setini yükleme
    print("Stable Bias Professions veri seti indiriliyor...")
    bias_dataset = load_dataset("society-ethics/stable-bias-professions")

    # Eğitim verilerini CSV dosyasına kaydetme
    bias_train = bias_dataset['train']
    bias_train_df = bias_train.to_pandas()
    bias_output_path = "C:\\Users\\User\\Desktop\\stable_bias_professions_train.csv"

    bias_train_df.to_csv(bias_output_path, index=False)
    print(f"Stable Bias Professions veri seti eğitim verileri {bias_output_path} dosyasına kaydedildi.")

    # Veri seti analizi
    print(bias_train_df.info())
    print(bias_train_df.head())
    print(bias_train_df.describe())

    # Eksik değerlerin kontrolü ve temizlenmesi
    if bias_train_df.isnull().sum().any():
        bias_train_df = bias_train_df.dropna()
        print("Eksik değerler temizlendi.")

    # Sınıf dengesizliği analizi
    print("Sınıf dağılımı:")
    print(Counter(bias_train_df['profession']))

    # Transformer tabanlı model yükleme
    print("Transformer tabanlı model yükleniyor...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Veri setini eğitim ve değerlendirme olarak ayır
    train_data, eval_data = train_test_split(bias_dataset["train"], test_size=0.2, random_state=42)

    # Örnek bir eğitim döngüsü
    print("Model eğitimi için hazırlanıyor...")
    def preprocess_function(examples):
        return tokenizer(examples["profession"], truncation=True, padding=True, max_length=128)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_eval = eval_data.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="C:\\Users\\User\\Desktop\\trainer_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="C:\\Users\\User\\Desktop\\logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,  # Değerlendirme veri seti
        tokenizer=tokenizer,
    )

    # Model eğitimi
    print("Model eğitiliyor...")
    trainer.train()

    # Örnek metinlerin etik değerlendirmesi
    print("Tahminler oluşturuluyor...")
    texts = bias_train_df['profession'][:10].tolist()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).tolist()

    print("Transformer tabanlı model tahminleri:")
    for text, pred in zip(texts, predictions):
        print(f"Meslek: {text} -> Tahmin: {'Etik' if pred == 1 else 'Etik Değil'}")

    # Metrik hesaplamaları (örnek)
    true_labels = [1, 0, 1, 1, 0]  # Gerçek etiketler
    predicted_labels = [1, 0, 0, 1, 1]  # Tahmin edilen etiketler

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    # Metrik sonuçlarını yazdır
    print("Metrikler:")
    print(f"Doğruluk (Accuracy): {accuracy}")
    print(f"Kesinlik (Precision): {precision}")
    print(f"Duyarlılık (Recall): {recall}")
    print(f"F1 Skoru: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"Cohen's Kappa: {kappa}")

    # Detaylı rapor
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    print("Sınıflandırma Raporu:")
    print(report)

    # Görseller oluşturma

    # Karışıklık Matrisi
    ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels, cmap='viridis')
    plt.title("Confusion Matrix")
    plt.savefig("C:\\Users\\User\\Desktop\\confusion_matrix.png")
    plt.show()

    # ROC Eğrisi
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("C:\\Users\\User\\Desktop\\roc_curve.png")
    plt.show()

    # Bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'MCC', "Cohen's Kappa"]
    values = [accuracy, precision, recall, f1, roc_auc, mcc, kappa]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=values, palette="viridis")
    plt.title("Sınıflandırma Metrikleri")
    plt.ylabel("Değer")
    plt.ylim(0, 1)
    plt.savefig("C:\\Users\\User\\Desktop\\metrics_plot.png")
    plt.show()

except Exception as ex:
    print("Bir hata oluştu:", ex)
