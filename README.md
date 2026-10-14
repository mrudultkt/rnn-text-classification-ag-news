# AG News Text Classification: Simple RNN vs LSTM vs GRU vs Bi-LSTM

A deep learning project that classifies news articles from the AG News dataset into four topics using recurrent neural networks. It builds and compares four sequence models (Simple RNN, LSTM, GRU and Bi-LSTM) under identical settings, and also studies the effect of gradient clipping on training stability.

## Project Overview

The objective is to understand how different recurrent architectures perform on a text classification task. All models use the same preprocessing, data splits, optimizer, loss, epochs and batch size, so the comparison stays fair and only the recurrent layer changes.

The study answers five questions:

- Which model achieves the best test accuracy?
- Do LSTM and GRU outperform the Simple RNN?
- Does reading the text in both directions (Bi-LSTM) improve performance?
- Which model trains fastest?
- Does gradient clipping make training more stable?

## Dataset

**AG News** is a news topic classification dataset, loaded through the Hugging Face `datasets` library. Each record contains a short news headline with its description, and a label from four categories.

| Label | Category |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

| Split | Articles |
|---|---|
| Training | 120,000 |
| Test | 7,600 |

The dataset is perfectly balanced: each class has 30,000 training and 1,900 test articles (25% each), so accuracy is a fair evaluation metric. The training set was further split with a stratified 80/20 split into **96,000 training** and **24,000 validation** articles. The test set was kept separate for final evaluation.

## Technologies Used

- Python
- Jupyter Notebook (Google Colab / Kaggle)
- TensorFlow / Keras
- Hugging Face Datasets
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Workflow

1. Load and explore the dataset (structure, sample articles, class distribution)
2. Clean leftover HTML codes in the text
3. Tokenize the text with a 20,000-word vocabulary and an out-of-vocabulary token
4. Convert articles to integer sequences and pad them to a fixed length of 100
5. Create a stratified validation set
6. Build, train and evaluate a Simple RNN
7. Build, train and evaluate an LSTM
8. Build, train and evaluate a GRU
9. Build, train and evaluate a Bidirectional LSTM
10. Train the Simple RNN with and without gradient clipping
11. Compare all models and write the final conclusion

## Preprocessing

| Setting | Value |
|---|---|
| Vocabulary size | 20,000 words (rare words mapped to an `<OOV>` token) |
| Sequence length | 100 words |
| Padding | Zeros added at the start (pre-padding) |
| Truncation | Long articles cut at the end |
| Labels | Integers 0 to 3 |

Articles are short (average 38 words), and a length of 100 keeps 99.82% of them intact. The tokenizer was fitted on the training text only, so no information from the test set leaks into the vocabulary.

## Model Architectures

All four models share the same structure, and only the recurrent layer changes:

```
Embedding(20000, 64) -> [Recurrent layer, 64 units] -> Dense(64, relu) -> Dense(4, softmax)
```

| Model | Recurrent layer | Layer parameters | Total parameters |
|---|---|---:|---:|
| Simple RNN | `SimpleRNN(64)` | 8,256 | 1,292,676 |
| LSTM | `LSTM(64)` | 33,024 | 1,317,444 |
| GRU | `GRU(64)` | 24,960 | 1,309,380 |
| Bi-LSTM | `Bidirectional(LSTM(64))` | 66,048 | 1,354,564 |

## Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam (default learning rate) |
| Loss | `sparse_categorical_crossentropy` |
| Metric | Accuracy |
| Epochs | 10 |
| Batch size | 128 |
| Random seed | 42 |
| Hardware | Kaggle GPU (NVIDIA Tesla T4) |

## Results

| Model | Training Acc | Validation Acc | **Test Acc** | Test Loss | Training Time |
|---|---:|---:|---:|---:|---:|
| Simple RNN | 98.89% | 85.97% | 86.01% | 0.7279 | 81.0 s |
| LSTM | 98.58% | 89.94% | **89.38%** | 0.5757 | 69.5 s |
| GRU | 98.75% | 89.01% | 88.89% | 0.6271 | **65.6 s** |
| Bi-LSTM | 98.55% | 88.87% | 88.87% | **0.4906** | 95.8 s |

Training and validation accuracy are the values from the final epoch (epoch 10).

### Best Validation Accuracy per Model

| Model | Best validation accuracy | Epoch |
|---|---:|---:|
| Simple RNN | 89.75% | 1 |
| LSTM | 91.40% | 1 |
| GRU | 91.30% | 2 |
| Bi-LSTM | 91.59% | 1 |

### Gradient Clipping Experiment (Simple RNN)

| Measure | Without clipping | With clipping (`global_clipnorm=1.0`) |
|---|---:|---:|
| Training accuracy | 98.92% | 98.61% |
| Validation accuracy | 87.77% | 86.69% |
| Test accuracy | 87.74% | 86.83% |
| Test loss | 0.6131 | 0.6561 |
| Training loss after epoch 1 | 0.492 | 0.685 |
| Largest raw gradient norm | 4.99 | 4.23 |
| Training time | 84.8 s | 84.9 s |

## Key Findings

- The **LSTM** achieved the best test accuracy (89.38%), followed by the GRU (88.89%), the Bi-LSTM (88.87%) and the Simple RNN (86.01%).
- Both gated models clearly outperformed the Simple RNN, with lower test loss, a smaller training-validation gap and higher validation accuracy at every epoch. The Simple RNN had the largest gap (12.9 points, compared with 8.6 for the LSTM).
- The **Bi-LSTM did not improve accuracy** over the regular LSTM and was the slowest model (about 38% longer), although it achieved the lowest test loss. News topics are mostly identified by key words, which a forward-only model can already capture, and the zero padding at the start of each article may reduce what the backward direction contributes.
- The **GRU trained fastest** and was within 0.5 points of the LSTM, with about 24% fewer recurrent parameters.
- **All models overfit.** Each reached its best validation accuracy in epoch 1 or 2 (about 90% to 91.6%), after which validation loss rose steadily until epoch 10.
- **Gradient clipping did not noticeably improve stability.** No exploding gradients occurred (no `NaN` values or loss spikes), although the raw gradient norm was usually above the clip threshold of 1.0. Clipping slowed early learning and did not improve validation behaviour or test accuracy.
- Results vary slightly between runs. The same Simple RNN configuration gave 86.01% and 87.74% test accuracy in two runs, so differences below about 1 point (such as LSTM vs GRU vs Bi-LSTM) should not be over-interpreted.

## Final Conclusion

The LSTM is the recommended model for this dataset. It achieved the highest test accuracy and the smallest overfitting gap at a modest training cost, while the GRU is an equally practical, slightly faster alternative. The Simple RNN was clearly the weakest, and the Bi-LSTM was not worth its extra cost, because reading the article in both directions did not improve accuracy. Gradient clipping was not needed with these settings. A useful next step would be to add early stopping and dropout, since all models peaked early and then overfitted.

## Repository Structure

```
├── ipynb file
└── README.md
```

## How to Run

This project trains four neural networks on 120,000 articles, so a **GPU runtime is recommended**. Run it on Google Colab or Kaggle.

### Option 1: Google Colab (GPU)

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**, then upload the `.ipynb` file from this repository.
2. Choose **Runtime → Change runtime type**, select **T4 GPU** and click **Save**.
3. Run the first cell, which installs the `datasets` library and prints the available GPUs. Confirm that a GPU device is listed.
4. Choose **Runtime → Run all**, or run the cells from top to bottom.

### Option 2: Kaggle (GPU)

1. Open [Kaggle](https://www.kaggle.com/), click **Create → New Notebook**, then choose **File → Import Notebook** and upload the `.ipynb` file.
2. In the right-hand **Settings** panel, set **Accelerator** to a GPU (for example GPU T4 x2).
3. In the same panel, turn **Internet** on. This is required to download the dataset