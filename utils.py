import glob
from typing import List
import pickle
from music21 import converter, instrument, chord
from music21.note import Note
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
import config as cfg


def load_songs(songs_filepath: str) -> List[List[str]]:
    all_file_notes = list()

    for file in glob.glob(songs_filepath):
        print(f"Loading file {file}", end=" ")

        midi = converter.parse(file)
        midi_partitioned_by_instrument = instrument.partitionByInstrument(midi)
        file_notes = midi_partitioned_by_instrument.parts[0].recurse()

        notes = list()

        for element in file_notes:
            if isinstance(element, Note):
                notes.append(str(element.pitch))
                pass
            elif isinstance(element, chord.Chord):
                notes.append('.'.join([str(note.pitch) for note in element.notes]))

        all_file_notes.append(notes)

        print(f"finished {len(notes)}")
    return all_file_notes


def serialize_notes(notes: List[List[str]]) -> None:
    with open(cfg.midi_songs_serialized_path, "wb") as file:
        pickle.dump(notes, file)


def deserialize_notes() -> List[List[str]]:
    with open(cfg.midi_songs_serialized_path, "rb") as file:
        return pickle.load(file)


def prepare_sequences(all_notes: List[List[str]]):
    sequence_length = cfg.sequence_length
    sequences = list()
    sequences_out = list()

    for notes in all_notes:
        if len(notes) <= sequence_length:
            raise Exception("Too short sequence")
        for i in range(0, len(notes) - sequence_length):
            sequences.append(notes[i:i + sequence_length])
            sequences_out.append(notes[i + sequence_length])

    return sequences, sequences_out


def prepare_input_output(sequences_in, sequences_out):
    all_pitch_names = set([note for sequence in sequences_in for note in sequence] + [note for note in sequences_out])
    pitch_names = sorted(all_pitch_names)
    pitch_amount = len(pitch_names)
    notes_map = dict((note, number) for number, note in enumerate(pitch_names))

    n_patterns = len(sequences_in)

    network_in = [[notes_map[note] for note in seq_in] for seq_in in sequences_in]
    network_out = [notes_map[note] for note in sequences_out]

    normalized_in = np.reshape(network_in, (n_patterns, cfg.sequence_length, 1)) / float(pitch_amount)

    network_out = np_utils.to_categorical(network_out)

    return network_in, normalized_in, network_out, pitch_names


def create_network_model(network_input, n_vocab):
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

    model.summary()

    return model
