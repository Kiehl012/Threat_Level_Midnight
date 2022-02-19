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
    /api/v1.0/episode/<ep start>/<ep end>
    ''')


@app.route("/api/v1.0/actor", methods=('GET'))
@app.route("/api/v1.0/actor/start", methods=('GET')
@app.route("/api/v1.0/actor/start/end") 
def actor_lookup(actor, start=None, end=None):
    if not start:
        results = session.query.filter(Scripts.emp_name == actor).all()
    elif not end:
        results = session.query.filter(Scripts.emp_name == actor).\
                  filter(Scripts.episode_id >= start).all()
    else:
        results = session.query.filter(Scripts.emp_name == actor).\
                  filter(Scripts.episode_id >= start).\
                  filter(Scripts.episode_id <= end).all()
    return jsonify(results)


@app.route("/api/v1.0/episode/start", methods=('GET'))
@app.route("/api/v1.0/episode/start/end", methods=('GET'))
def episode_lookup(start=None, end=None)
    if not start:
        result = {}
    elif not end:
        results = session.query.filter(Scripts.episode_id = start).all()
    else:
        results = session.query.filter(Scripts.episode_id >= start).\
                  filter(Scripts.episode_id < = end).all()
    return jsonify(results)


@app.route("/", methods=('GET'))
def index():
    results = {} 
    return jsonify(reults)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5090, debug=True)
