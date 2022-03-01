from flask import Flask, render_template, jsonify, request
from flask import request
import tensorflow as tf
import numpy as np

one_step_reloaded = tf.saved_model.load('../Text_Gen_Model/RNN_Model')

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/prediction_route', methods=['GET','POST'])
def prediction_post():
      
    if request.method == 'POST':        
        user_txt = request.form["michael_talks"]
        next_char = tf.constant([user_txt])        
        states = None    
        result = [next_char]
        start_txt = len(next_char[0].numpy().decode('utf-8'))
        punct = ['!','?','.']
        punct_counter = 0

        for n in range(150):
            next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
            if next_char in punct:
                punct_counter += 1
                if (punct_counter > 2) and ((n+start_txt) > 80):
                    result.append(next_char)
                    print(next_char[0].numpy().decode('utf-8'))
                    break
            result.append(next_char)

        result = tf.strings.join(result)
        prediction = result[0].numpy().decode('utf-8')        
        
        return render_template("index.html", prediction=prediction)
        # return render_template("index.html", prediction=user_txt)

if __name__ == "__main__":
    app.run(debug=True)

