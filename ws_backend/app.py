import datetime
import os

from flask import Flask, requests, jsonify

from models import Book 
from database import db_session

app = Flask(__name__)
app.secret_key = os.environ['APP_SECRET_KEY']

@app.route("/api/v1.0/help", methods=('GET'))
def help():
    return(
    '''
    Welcome to the Oreilly Book API!
    Available Routes:
    /
    /api/v1.0/help
    /api/v1.0/title/<book title>
    /api/v1.0/bookid/start/end
    /api/v1.0/add_book/
    ''')


@app.route("/api/v1.0/title", methods=('GET'))
def title_search():
    title = requests.args.get('title')
    print(title)
    results = Book.query.filter(Book.title.match(%title%))
    return jsonify(results)

@app.route("/api/v1.0/bookid/start")
@app.route("/api/v1.0/bookid/start/end") 
def bookid_lookup(start=None, end=None):
    if not end:
        results = Book.query(id=start)
    else:
        results = session.query.\
        filter(Book.id >= start).\
        filter(Book.id <= end).all()
    return jsonify(results)

@app.route("/api/v1.0/add_book", methods=('POST'))
def add_book():
    return 0

@app.route("/", methods=('GET'))
def index():
    results = Book.query.all()
    return jsonify(reults)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5090, debug=True)
