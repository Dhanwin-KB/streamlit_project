import streamlit as lit
import pandas as pd

lit.set_page_config(
    page_title= "About"
)

def main():
    lit.title("Meta's AudioCraft")
    lit.write("AudioCraft represents a leap forward in the realm of generative AI for audio, offering a versatile and comprehensive codebase for a wide range of audio needs. Their suite comprises three distinct models—MusicGen, AudioGen, and EnCodec. Out of which MusicGen was used for this project.")
    lit.write("MusicGen’s generative AI model empowers users to generate music from scratch. By providing textual input, users can prompt MusicGen to craft musical pieces that resonate with their artistic vision. Musicians can collaborate with MusicGen to explore uncharted genres which are beyond their own style and expertise.")

if __name__ == "__main__":
    main()
