#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import pathlib

mike_model = tf.saved_model.load('michael_rnn_model')
jim_model = tf.saved_model.load('jim_rnn_model')
pam_model = tf.saved_model.load('pam_rnn_model')
dwight_model = tf.saved_model.load('dwight_rnn_model')
andy_model = tf.saved_model.load('andy_rnn_model')

start = time.time()
mike_states = None
mike_nchar = tf.constant(["Dwight,"])
dwight_states = jim_states = pam_states = andy_states = None
dwight_nchar = jim_nchar = pam_nchar = andy_nchar = tf.constant(["Michael..."])


for lines in range(30):
    mike_result = []
    dwight_result = []
    jim_result = []
    pam_result = []
    andy_result = []

    for n in range(120):
        # Michael's stuff
        mike_nchar, mike_states = mike_model.\
                                  generate_one_step(mike_nchar,
                                                    states=mike_states)
        mike_result.append(mike_nchar)
    mike_loop_result = tf.strings.join(mike_result)

    # kick off Dwight using what Mike last said
    dwight_nchar, dwight_states = dwight_model.\
                                  generate_one_step(mike_loop_result,
                                                    states=dwight_states)
    for n in range(120):
        dwight_nchar, dwight_states = dwight_model.\
                                      generate_one_step(dwight_nchar,
                                                        states=dwight_states) 
        dwight_result.append(dwight_nchar)
    dwight_loop_result = tf.strings.join(dwight_result)

    # kick off Jim using what Dwight last said
    jim_nchar, jim_states = jim_model.\
                            generate_one_step(dwight_loop_result,
                                              states=jim_states)
    for n in range(100):
        jim_nchar, jim_states = jim_model.generate_one_step(jim_nchar,
                                                            states=jim_states)
        jim_result.append(jim_nchar)
    jim_loop_result = tf.strings.join(jim_result)

    # kick off Pam using what Jim last said
    pam_nchar, pam_states = pam_model.generate_one_step(jim_loop_result,
                                                        states=pam_states)
    for n in range(100):
        pam_nchar, pam_states = pam_model.generate_one_step(pam_nchar,
                                                            states=pam_states)
        # andy does his own thing here
        andy_nchar, andy_states = andy_model.generate_one_step(andy_nchar,
                                                               states=andy_states)
        pam_result.append(pam_nchar)
        andy_result.append(andy_nchar)

    pam_loop_result = tf.strings.join(pam_result)
    andy_loop_result = tf.strings.join(andy_result)

    end = time.time()

    char_speech = {}
    char_speech['michael'] = mike_loop_result
    char_speech['dwight'] = dwight_loop_result
    char_speech['jim'] = jim_loop_result
    char_speech['pam'] = pam_loop_result
    char_speech['andy'] = andy_loop_result

    print('\n')
    for char, result in char_speech.items():
        punct = ['!', '?', '.']
        text_output = result[0].numpy().decode('utf-8')
        text_output.lstrip()
        text_output.rstrip()
        rtext = text_output[::-1]
        for p in punct:
            if not rtext.startswith(p):
                rtext = rtext[1::]
            else:
                break
        final_text = rtext[::-1]

        print(char, ': ', final_text)
        #print(char, ': ', result[0].numpy().decode('utf-8'), '\n')
        #print('\n')
        result = None
    print('Run time:', end - start, '\n')

