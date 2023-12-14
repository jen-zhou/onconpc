from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to my Flask App!"

if __name__ == "__main__":
    app.run(port=5000)