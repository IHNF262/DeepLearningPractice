from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import IPython
import sys
from music21 import *
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *

IPython.display.Audio('./data/30s_seq.mp3')
