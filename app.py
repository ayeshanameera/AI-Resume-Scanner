from flask import Flask, request, render_template
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    if request.method == 'POST':
        resume = docx2txt.process(request.files['resume'])
        job_desc = request.form['job_desc']
        cv = CountVectorizer()
        count_matrix = cv.fit_transform([resume, job_desc])
        score = round(cosine_similarity(count_matrix)[0][1] * 100, 2)
    return render_template('index.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)
