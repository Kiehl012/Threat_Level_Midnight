#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import os
import time
import pathlib


# In[6]:


one_step_reloaded = tf.saved_model.load('RNN_Model')


# In[10]:


start = time.time()
states = None
next_char = tf.constant(["Dwight,"])
result = [next_char]

for n in range(105):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

