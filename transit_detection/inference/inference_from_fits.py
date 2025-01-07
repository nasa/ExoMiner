from keras.saving import load_model
from pathlib import Path


if __name__ == '__main__':
    MODEL_PATH = Path('path/to/file')

    model = load_model(MODEL_PATH)
    