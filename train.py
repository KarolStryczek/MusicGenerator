import glob
import numpy as np
from typing import List
from music21 import converter, instrument, chord
from music21.note import Note
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint
from properties import properties
import utils
import config as cfg


def run() -> None:
    all_notes = utils.load_songs(cfg.config["midi_songs_path"])
    utils.serialize_object(all_notes,cfg.config["midi_songs_serialized_path"])
    sequences_in, sequences_out = prepare_sequences(all_notes)
    network_in, network_out, pitch_amount = prepare_input_output(sequences_in, sequences_out)
    model = create_network(network_in, pitch_amount)
    train(model, network_in, network_out)


def prepare_sequences(all_notes: List[List[str]]):
    sequence_length = properties['sequence_length']
    sequences = list()
    sequences_out = list()

    for notes in all_notes:
        if len(notes) <= sequence_length:
            raise Exception("I think your sequence should be a little bit longer")
        for i in range(0, len(notes) - sequence_length):
            sequences.append(notes[i:i + sequence_length])
            sequences_out.append(notes[i + sequence_length])

    return sequences, sequences_out


def prepare_input_output(sequences_in, sequences_out):
    pitchnames = sorted(set([note for sequence in sequences_in for note in sequence] + [note for note in sequences_out]))
    pitch_amount = len(pitchnames)
    notes_map = dict((note, number) for number, note in enumerate(pitchnames))

    n_patterns = len(sequences_in)

    network_in = [[notes_map[note] for note in seq_in] for seq_in in sequences_in]
    network_out = [notes_map[note] for note in sequences_out]

    network_in = np.reshape(network_in, (n_patterns, properties['sequence_length'], 1))

    network_in = network_in / float(pitch_amount)

    network_out = np_utils.to_categorical(network_out)

    return network_in, network_out, pitch_amount


def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=50, batch_size=1024, callbacks=callbacks_list)


if __name__ == "__main__":
    run()