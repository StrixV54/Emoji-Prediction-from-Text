from flask import Flask, request, jsonify
from flask_cors import CORS
from flair.models import TextClassifier
from flair.data import Sentence
from flask import render_template
classifier = TextClassifier.load('./model/best-model.pt')

mapping = {
    'sad': '&#128546;',
    'smile': '&#x1F600',
    'food': '&#127869;&#65039;',
    'heart': '&#10084;&#65039;',
    'baseball': '&#x26be;',
    'afraid': "&#128561;",
    'angry': "&#128530",
    'excited': "&#129321;",
    'depressed': "&#128534;",
    'hate': "&#128545;"
}
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('emoji.html')


@app.route('/emojify', methods=['POST'])
def emoji():
    data = request.form.get('text')
    if not len(data.strip()):
        return ''
    sentence = Sentence(data)
    classifier.predict(sentence)
    print(str(sentence.labels))
    if 'sad' in str(sentence.labels):
        return mapping['sad']
    elif 'smile' in str(sentence.labels):
        return mapping['smile']
    elif 'food' in str(sentence.labels):
        return mapping['food']
    elif 'afraid' in str(sentence.labels):
        return mapping['afraid']
    elif 'depressed' in str(sentence.labels):
        return mapping['depressed']
    elif 'hate' in str(sentence.labels):
        return mapping['hate']
    elif 'angry' in str(sentence.labels):
        return mapping['angry']
    elif 'heart' in str(sentence.labels):
        return mapping['heart']
    elif 'excited' in str(sentence.labels):
        return mapping['excited']
    elif 'baseball' in str(sentence.labels):
        return mapping['baseball']


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=True)
