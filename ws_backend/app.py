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
    Welcome to the Threat Level Midnight API!
    Available Routes:
    /
    /api/v1.0/help
    /api/v1.0/actor/<actor name>
    /api/v1.0/actor/<actor name>/<episode>
    /api/v1.0/actor/<actor name>/<episode start>/<episode end>
    /api/v1.0/episode/<ep number>
    /api/v1.0/episode/start/end/
    ''')


@app.route("/api/v1.0/actor", methods=('GET'))
def actor_search():
    results = {} 
    return jsonify(results)

@app.route("/api/v1.0/actor/episodenum")
@app.route("/api/v1.0/actor/episode_start/episode_end") 
def actor_lookup(start=None, end=None):
    results = {}
    if not end:
        results
    else:
        #results = session.query.\
        #filter(Book.id >= start).\
        #filter(Book.id <= end).all()
        results
    return jsonify(results)

@app.route("/", methods=('GET'))
def index():
    results = {} 
    return jsonify(reults)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5090, debug=True)
