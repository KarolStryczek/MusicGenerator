from keras.callbacks import ModelCheckpoint
import utils
import config as cfg


def run() -> None:
    all_notes = utils.load_songs(cfg.midi_songs_path)
    utils.serialize_notes(all_notes)
    sequences_in, sequences_out = utils.prepare_sequences(all_notes)
    network_in, normalized_in, network_out, pitch_names = utils.prepare_input_output(sequences_in, sequences_out)
    model = utils.create_network_model(normalized_in, len(pitch_names))
    if cfg.continue_training:
        model.load_weights('weights.hdf5')
    train(model, normalized_in, network_out)


def train(model, network_in, network_out):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    model.fit(network_in, network_out, epochs=cfg.epochs_amount, batch_size=cfg.batch_size, callbacks=[checkpoint])


if __name__ == "__main__":
    run()
