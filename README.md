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
```bash
I am cold.    Je suis froid.
You are happy.    Tu es heureux.
```
Training: To train the model, you can use the train() function, which accepts the train_dataloader (returned by the get_dataloader() function) along with the encoder, decoder, and hyperparameters such as the number of epochs, learning rate, etc.

Evaluate the Model: After training, you can evaluate the model's performance on random sentence pairs using evaluateRandomly().

Attention Visualization: Use the evaluateAndShowAttention() function to evaluate a sentence and visualize the attention weights.
Example Usage:
```bash
# Prepare data
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

# Define encoder and decoder models
encoder = EncoderModel(input_lang.n_words, hidden_size=256)
decoder = DecoderModel(output_lang.n_words, hidden_size=256)

# Get the training dataloader
train_dataloader = get_dataloader(batch_size=64)

# Train the model
train(train_dataloader, encoder, decoder, n_epochs=10)

# Evaluate a random sentence
evaluateRandomly(encoder, decoder, n=5)

# Show attention visualization
evaluateAndShowAttention("I am happy", encoder, decoder)
```
Model Architecture
This implementation uses a simple encoder-decoder architecture with attention. The encoder processes the input sentence word-by-word and outputs a context vector. The decoder generates the target sentence word-by-word, using attention to focus on different parts of the input sequence.

Encoder
The encoder is typically an RNN (e.g., LSTM or GRU). It takes an input sequence, processes it step by step, and outputs a context vector that summarizes the information of the sequence.

Decoder
The decoder is another RNN, which takes the context vector and generates the output sequence, one word at a time. It uses attention to dynamically focus on different parts of the input sequence during decoding.

Training Loss
The training process minimizes the negative log likelihood (NLL) loss, which is commonly used for sequence generation tasks.

Visualizations
The code includes a function to visualize the loss during training (showPlot()) and the attention weights during translation (showAttention()).

Notes
CUDA Support: The code supports GPU training if CUDA is available.
Data Size: The model is designed for small datasets, with a maximum sentence length (MAX_LENGTH) of 10 tokens.
Customization: You can change the languages and corpus by modifying the dataset and adjusting the prepareData() function.
