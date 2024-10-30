from flask import Flask, request, jsonify, render_template
from llm import QA, ConversationBufferMemory  # Ensure this imports correctly based on your directory structure

app = Flask(__name__)

# Initialize memory
memory = ConversationBufferMemory()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    response = QA(question,memory)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
