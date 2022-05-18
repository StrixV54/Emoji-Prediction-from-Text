# Text2Emoji :relaxed:

![Emoji Representative](https://twistedsifter.files.wordpress.com/2015/06/the-best-text-emoticons-on-a-single-page.jpg)
Emoji-Prediction-From-Text is a text classification model trained on 200 sentences across 5 classes, using sentiment analysis. Below fig. shows the list of emoji's on which the model is trained on. Text to the side represents a high-level emotion that the emoji depicts. :sunglasses:

## Steps to Run

Note:Used Python 3.7 specifically.
1.Train the model locally. 2. `python main.py` 2. Open emoji.html in the browser and start typing :speech_balloon:

<!--
## Demo
![Text2Emoji Demo](https://github.com/StrixV54/Emoji-Prediction-from-Text/blob/master/demo.gif)
- -->

## Methodology

![Flair Internal](https://github.com/StrixV54/Emoji-Prediction-from-Text/assets/flair_internal1.png)

## Model Architecture

```
Model: "TextClassifier(
  (document_embeddings): DocumentRNNEmbeddings(
    (embeddings): StackedEmbeddings(
      (list_embedding_0): WordEmbeddings('glove')
      (list_embedding_1): FlairEmbeddings(
        (lm): LanguageModel(
          (drop): Dropout(p=0.05, inplace=False)
          (encoder): Embedding(300, 100)
          (rnn): LSTM(100, 2048)
          (decoder): Linear(in_features=2048, out_features=300, bias=True)
        )
      )
      (list_embedding_2): FlairEmbeddings(
        (lm): LanguageModel(
          (drop): Dropout(p=0.05, inplace=False)
          (encoder): Embedding(300, 100)
          (rnn): LSTM(100, 2048)
          (decoder): Linear(in_features=2048, out_features=300, bias=True)
        )
      )
      (list_embedding_3): ELMoEmbeddings(model=elmo-medium)
    )
    (word_reprojection_map): Linear(in_features=5732, out_features=256, bias=True)
    (rnn): LSTM(256, 512)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Linear(in_features=512, out_features=5, bias=True)
  (loss_function): CrossEntropyLoss()
)"
```

## Technologies Used

1. Flask
2. Flair
3. HTML/Bootstrap
4. Js
