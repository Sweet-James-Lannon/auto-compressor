from flask import Flask, render_template
import os

app = Flask(__name__, template_folder='dashboard')

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)



