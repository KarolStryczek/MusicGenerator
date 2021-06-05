import numpy
from music21 import instrument, stream
from music21.note import Note
from music21.chord import Chord
import utils
import config as cfg
from datetime import datetime
import random


def run():
    notes = utils.deserialize_notes()
    seq_in, seq_out = utils.prepare_sequences(notes)
    network_input, normalized_input, network_output, pitch_names = utils.prepare_input_output(seq_in, seq_out)
    model = utils.create_network_model(normalized_input, len(pitch_names))
    model.load_weights('weights.hdf5')
    for i in range(cfg.batch_generation_size):
        generate_song(model, network_input, pitch_names)


def generate_song(model, network_input, pitch_names):
    create_midi(generate_notes(model, network_input, pitch_names))


def generate_notes(model, network_input, pitch_names):
    notes_map = dict(enumerate(pitch_names))
    input_pattern = random.choice(network_input)
    # input_pattern = [random.randint(0, n_vocab) for i in range(cfg.sequence_length)]

    generated_notes = []
    for note_index in range(cfg.output_length):
        print(f'generating: {note_index}')
        prediction_input = numpy.reshape(input_pattern, (1, len(input_pattern), 1)) / float(len(pitch_names))

        prediction = model.predict(prediction_input, verbose=0, batch_size=cfg.batch_size)

        index = numpy.argmax(prediction)
        generated_notes.append(notes_map[index])

        input_pattern.append(index)
        input_pattern = input_pattern[1:]

    return generated_notes


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for item in prediction_output:
        if '.' in item:
            output_notes.append(parseChord(chord_str=item, offset=offset))
        else:
            output_notes.append(parseNote(note_str=item, offset=offset))
        offset += 0.5

    stream.Stream(output_notes).write('midi', fp=f'results/out_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mid')


def parseChord(chord_str: str, offset: float) -> Chord:
    notes = [parseNote(note) for note in chord_str.split('.')]
    chord = Chord(notes)
    chord.offset = offset
    return chord


def parseNote(note_str: str, offset: float = None) -> Note:
    note = Note(note_str)
    note.storedInstrument = instrument.Piano()
    if offset is not None:
        note.offset = offset
    return note


if __name__ == '__main__':
    run()
