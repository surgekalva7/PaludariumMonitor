from flask import Flask, render_template

app = Flask(__name__)

# Routes for the web application
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)