import glob
from typing import List
import pickle
from music21 import converter, instrument, chord
from music21.note import Note

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


def serialize_object(object_to_be_serialized: object, filepath: str) -> None:
    with open(filepath, "wb") as file:
            pickle.dump(object_to_be_serialized, file)
    return None


def read_serialized_object(filepath: str) -> object:
    with open(filepath, "rb") as file:
        unserialized = pickle.load(file)
    return unserialized
    
