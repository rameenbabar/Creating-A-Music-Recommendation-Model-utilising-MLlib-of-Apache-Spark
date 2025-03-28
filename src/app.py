from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    # List all mp3 files in the static/audio directory
    audio_files = [f for f in os.listdir('static/audio') if f.endswith('.mp3')]
    audio_files = ['audio/' + file for file in audio_files]  # Prepend the directory path
    return render_template('index.html', audio_files=audio_files)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
