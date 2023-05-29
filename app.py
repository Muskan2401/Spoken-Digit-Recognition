import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from PIL import Image
 

def main():
  

    st.title("Spoken Digit Recognition")
    selectbox = st.sidebar.selectbox('Select the type of input', ('None','Upload Audio'))
    
    if selectbox == 'Upload Audio':
        audio_file = st.file_uploader("Upload Audio", type=['wav'])
        if audio_file is not None:
            frame_rate, data = wav.read(audio_file)
            signal_wave = wave.open(audio_file)
            sig = np.frombuffer(signal_wave.readframes(frame_rate), dtype=np.int16)
            plt.specgram(sig, NFFT=1024, Fs=frame_rate, noverlap=900)
            img=fig.savefig(f'{audio_file}.png', dpi=fig.dpi)
        if st.button("Recognise"):
            result_text= detect_audio(img)
            st.text(result_text)
        
    if selectbox == 'None':
        st.text("Choose options from sidebar->dropdown")



def detect_audio(img):

    modelSaved = keras.models.load_model('Spectogram.h5') 
    img = tf.keras.preprocessing.image.load_img("img", target_size=(432, 288))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = modelSaved.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    score
    result=np.argmax(score)

    return result
   

if __name__ == '__main__':
    main()