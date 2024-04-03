import streamlit as lit
import pandas as pd
from PIL import Image

lit.set_page_config(
    page_title= "Tutorial"
)

img1 = Image.open("tut1.jpg")
img2 = Image.open("tut2.jpg")
img3 = Image.open("tut3.jpg")
img4 = Image.open("tut4.jpg")

lit.title("Tutorial")
lit.write("1. Enter the text prompt for Music Generation : Users are expected to provide textual input to MusicGen, describing the musical piece they want to generate. This could include information about the genre, mood, tempo, instruments, and other musical characteristics.")
lit.image(img1)
lit.write("2. Enter the song duration (in seconds) : Users are expected to provide the expected duration of the output audio")
lit.image(img2)
lit.write("3. Generation : When the user submits the input, the app generates music based on the provided text prompt and selected duration. This is achieved by feeding the prompt into the pre-trained model.")
lit.image(img3)
lit.write("4. Audio Output: The generated music is then converted into audio samples and saved as a WAV file. Additionally, the audio is displayed on the web app, allowing users to listen to it directly.")
lit.image(img4)
lit.write("5. Download Option: Users are provided with a download link to save the generated audio file locally.")

lit.header('Prompt Examples :')
lit.write("\"A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.\"")
lit.write("\"Classic reggae track with an electronic guitar solo\"")
lit.write("\"Drum and bass beat with intense percussions\"")
lit.write("\"A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city.\"")
lit.write("\"Rock with saturated guitars, a heavy bass line and crazy drum break and fills.\"")