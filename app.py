import tensorflow as tf
import numpy as np

one_step_reloaded = tf.saved_model.load('./Text_Gen_Model/RNN_Model')

user_txt = input('Enter prompt here:')
states = None
next_char = tf.constant([user_txt])
result = [next_char]

for n in range(105):
    next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
prediction = result[0].numpy().decode('utf-8')
print(prediction)