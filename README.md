# Neural-Machine-Translation
Neural Machine Translation (NMT) with Seq2Seq Model
This repository contains a PyTorch implementation of a Neural Machine Translation (NMT) model using a Sequence-to-Sequence (Seq2Seq) architecture with attention. It is designed to translate sentences from English (eng) to French (fra).

Requirements
Make sure you have the following Python libraries installed:

torch (PyTorch)
numpy
matplotlib
re
unicodedata
You can install the necessary libraries using:
```bash
pip install torch numpy matplotlib
```
Overview
This code demonstrates how to use a Seq2Seq model to perform machine translation tasks. The main components of this code are:

- Data Preprocessing: Normalization of the input data (text pairs).
- Model Architecture: A Seq2Seq model with attention mechanism.
- Training: Training the model with mini-batches and computing loss.
- Evaluation: Evaluating the model by translating new sentences and visualizing the attention mechanism.
File Structure
- data/: Folder containing the dataset (in this case, an English-French parallel corpus, e.g., eng-fra.txt).
- main.py: Contains the implementation of the Seq2Seq model, data processing functions, training loop, and evaluation code.
Key Classes & Functions
Lang Class
The Lang class defines the language for the translation task, storing the word-to-index mapping, count of each word, and a reverse index mapping.

Methods:
- addSentence(sentence): Adds a sentence (splits it into words) to the language.
- addWord(word): Adds a word to the vocabulary.
Data Preprocessing
normalizeString(s)
Normalizes a string by converting it to ASCII, lowercasing, and removing any non-alphabetic characters.

readLangs(lang1, lang2, reverse=False)
Reads a parallel corpus file (data/{lang1}-{lang2}.txt), processes sentence pairs, and returns two Lang objects (for source and target languages) and the list of sentence pairs.

prepareData(lang1, lang2, reverse=False)
Prepares the data by reading the file, filtering out long or irrelevant sentence pairs, and counting the vocabulary.

Seq2Seq Model Components
- Encoder: Encodes the input sequence into a fixed-length context vector.
- Decoder: Decodes the context vector into the output sequence.
- Attention Mechanism: Uses the encoder's output to weight the importance of different words while decoding.
Training and Evaluation
train()
Trains the Seq2Seq model for a given number of epochs, using the provided batch size and learning rate. It prints the loss at regular intervals and visualizes the loss.

evaluate()
Evaluates the trained model by translating a given input sentence and comparing the output with the expected result.

evaluateRandomly()
Randomly selects pairs from the dataset and evaluates them, printing the model's translations.

showAttention()
Visualizes the attention weights during translation, showing which words the model focused on when generating each part of the output sequence.

Helper Functions
- asMinutes(s): Converts seconds to a "minutes:seconds" string format.
- timeSince(since, percent): Displays the elapsed time and the estimated remaining time during training.
- showPlot(points): Plots the training loss over time.
Usage
Prepare the Dataset: The dataset should be a tab-separated file with sentences in the source language on the left and the target language on the right. An example file format for eng-fra.txt is:
