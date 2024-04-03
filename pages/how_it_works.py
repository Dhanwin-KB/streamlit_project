import streamlit as lit
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

lit.set_page_config(
    page_title= "How it Works"
)

img1 = Image.open("1.jpg")
img2 = Image.open("2.jpg")

lit.header('How It Works')
lit.image(img1)
lit.write("MusicGen consists of a single autoregressive Language Model (LM) that operates over streams of compressed discrete music representation, i.e., tokens. A simple approach to leverage the internal structure of the parallel streams of tokens is introduced and with a single model and elegant token interleaving pattern, it efficiently models audio sequences, simultaneously capturing the long-term dependencies in the audio and allowing one to generate high-quality audio.")
lit.image(img2)
lit.write("Their models leverage the EnCodec neural audio codec to learn the discrete audio tokens from the raw waveform. EnCodec maps the audio signal to one or several parallel streams of discrete tokens. It then uses a single autoregressive language model to recursively model the audio tokens from EnCodec. The generated tokens are then fed to EnCodec decoder to map them back to the audio space and obtain the output waveform. Finally, different types of conditioning models can be used to control the generation such as using a pretrained text encoder for text-to-audio applications.")

lit.header('In Simple Terms')

lit.write("1. Receiving Text Input: Users provide textual input to MusicGen, describing the musical piece they want to generate. This could include information about the genre, mood, tempo, instruments, and other musical characteristics.")
lit.write("2. Encoding Text: The text input is encoded into a format that the model can understand and process. This encoding typically involves converting the text into a numerical representation that can be fed into the neural network model.")
lit.write("3. Comparison with Audio Tokens: MusicGen's neural network model has been trained on a dataset of audio samples, which are represented as tokens. These tokens capture various aspects of the audio, such as musical notes, rhythms, and other features.")
lit.write("4. Generating Audio Based on Matching Tokens: MusicGen compares the encoded text input with the audio tokens available from its model. It identifies tokens that closely match the characteristics described in the text input.")
lit.write("5. Generating Music: Based on the matched tokens and other contextual information, MusicGen generates a musical piece that reflects the user's input. This could involve composing melodies, arranging musical elements, and applying other musical techniques.")
lit.write("6. Output: The generated music is then provided as the output, which users can listen to, modify, or further refine as needed.")
lit.write("Overall, MusicGen leverages text-to-audio generation techniques, where textual input is translated into musical compositions through the use of neural networks trained on audio data. This process allows users to create music based on textual descriptions, expanding the possibilities for musical creativity and exploration.")

lit.header('Code and Explanation')

lit.write("Importing libraries and dependencies")
lit.code("from audiocraft.models import MusicGen\nimport streamlit as st\nimport torch\nimport torchaudio\nimport os\nimport numpy as np\nimport base64")
lit.write("""
audiocraft.models: as it contains classes, functions, and other components related to music generation.
\nstreamlit: for building the interactive web application.
\ntorch: for tensor computation (like NumPy) with strong GPU acceleration.
\ntorchaudio: for audio processing tasks - loading, transforming, and analyzing audio data.
\nos: for interacting with the operating system.
\nnumpy: for scientific computations.
\nbase64: for converting binary data (audio) into a format that can be easily transmitted over text-based protocols like HTTP.
""")

lit.header("load_model()")
lit.code("""@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model""")
lit.write("Using \'@st.cache_resource\', subsequent calls to the function with the same input parameters will return the cached result instead of recomputing it. This significantly speeds up the execution of the function. Fetching the model would take time and this ensures it isnt very time consuming since we're using the cached copy.")
lit.write("This function basically loads the model.It loads the pre-trained model called 'facebook/musicgen-small' using the get_pretrained method from the MusicGen class.It is a pre-trained music generation model provided by Facebook's MusicGen project.")

lit.header("generate_music_tensors()")
lit.code("""def generate_music_tensors(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]""")
lit.write("This function generates music tensors based on the user input prompt and duration.This line calls the load_model function to load a pre-trained music generation model.Here, it sets the generation parameters for the loaded model. It enables sampling, set a top-k value (for selecting tokens during generation), and specify the duration of the generated music.It generates music based on the given description using the loaded model. It provides the description, enables progress tracking during generation, and requests to return the generated tokens.Finally, it returns the generated output, which is expected to be a tensor representing the generated music.")

lit.header("save_audio()")
lit.code("""def save_audio(samples: torch.Tensor):
    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)""")
lit.write("This function is for rendering an audio player for the given audio samples and saving them to a local directory.It takes one parameter - samples which is a tensor containing the audio samples to be saved. The shape of the tensor should be [B, C, T] or [C, T], where B represents the batch dimension (if present), C represents the number of channels, and T represents the time steps.The sample rate and the directory path is set where the audio files will be saved.Line detaches the tensor from the computation graph and moves it to the CPU.It iterates over the audio samples and saves each one to a separate audio file in the specified directory. It constructs the file path based on the index of the sample and saves the audio using torchaudio.save, a function from the torchaudio library, specifying the sample rate as 32,000 Hz.")

lit.header("get_binary_file_downloader_html()")
lit.code("""def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href""")
lit.write("This function generates HTML code for creating a downloadable link to a binary file.It opens the binary file specified by bin_file in binary read mode ('rb'), reads its contents, and stores the data in the variable data.The binary data read from the file using base64 encoding. The b64encode function from the base64 library is used to perform the encoding. The resulting bytes-like object is then decoded into a UTF-8 string using the decode method.Finally, the function returns the HTML code for the download link. ")

lit.header("main()")
lit.code("""def main():

    st.title("Text to Music Generator")

    text_area = st.text_area("Enter your prompt")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if text_area and time_slider:
        st.json({
            'Your Prompt': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        print("Music Tensors: ", music_tensors)
        save_music_file = save_audio(music_tensors)
        audio_filepath = 'audio_output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()""")
lit.write("This function serves as the entry point for the Streamlit application.It displays the title at the top of the Streamlit application and create a text area and a slider widget. The text area accepts the prompt, and the slider allows users to select the duration of the generated music in seconds.It displays the user's input prompt and the selected time duration.It generates music tensors based on the user's input prompt and the selected time duration using the generate_music_tensors function. It saves the generated music as an audio file using the save_audio function. It displays the audio file using the Streamlit audio widget, allowing users to listen to the generated music.It also generates a downloadable link for the audio file using the get_binary_file_downloader_html function and displays it as Markdown. The link allows users to download the generated audio file.")