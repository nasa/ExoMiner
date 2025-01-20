from keras.saving import load_model
from pathlib import Path
import numpy as np

def preprocess_diff_img(image: np.ndarray):
    # apply dataset build preprocessing
    # apply norm
    pass

def preprocess_flux_window(flux_window: np.ndarray, resampling_rate: int=100):
    # apply dataset build preprocessing
    # apply norm
    pass

if __name__ == '__main__':
    #example input
    diff_img = np.random.rand(256, 256, 3)
    flux_window = np.random.rand(100)
    
    diff_img_tensor = preprocess_diff_img(diff_img)
    flux_window_tensor = preprocess_flux_window(flux_window)

    model = load_model("path_to_model.h5")

    #run inference
    predictions = model.predict([diff_img_tensor, flux_window_tensor])

    confidence_score = predictions[0][0]
    confidence_threshold = 0.5
    transit_detected = confidence_score > confidence_threshold

