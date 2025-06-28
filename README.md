# Federated Learning for Text Classification

This project demonstrates federated learning for text classification using the Reuters dataset and a bidirectional LSTM model in PyTorch. The goal is to collaboratively train NLP models across multiple simulated clients without sharing raw data.

## Features

- Federated learning setup with configurable number of clients
- Text preprocessing using NLTK and scikit-learn
- Bidirectional LSTM model for sequence classification
- Evaluation metrics: accuracy, precision, recall, and F1-score

## Dataset

- **Reuters newswire dataset** (downloaded via NLTK)
- Preprocessing: tokenization, stopword removal, vocabulary creation, label encoding, and padding

## Usage

1. Clone this repository.
2. Open the `Fed_NLP.ipynb` notebook in Google Colab or Jupyter Notebook.
3. Run all cells to:
   - Download and preprocess the dataset
   - Train and evaluate the model in a federated setup

## Requirements

- Python 3.x
- PyTorch
- NLTK
- scikit-learn
- Flower (`flwr`)
- pandas, numpy, matplotlib, tqdm

Install dependencies with:

```bash
pip install torch nltk scikit-learn flwr pandas numpy matplotlib tqdm
```

## Results

The notebook reports accuracy, precision, recall, and F1-score for the text classification task after training.

## License

This project is for educational purposes.
