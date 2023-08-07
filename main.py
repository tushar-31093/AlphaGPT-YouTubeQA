__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from PIL import Image
import whisper
import torch
import os
from streamlit_lottie import st_lottie 
from pytube import YouTube
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import pandas as pd
import requests

st.set_page_config(layout="centered", page_title="Youtube QnA")

#header of the application
# image = Image.open('logo.png')
 
 

 
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
url_lottie1 = "https://lottie.host/d860aaf2-a646-42f2-8a51-3efe3be59bf2/tpZB5YYkuT.json"
url_lottie2 = "https://lottie.host/93dcafc4-8531-4406-891c-89c28e4f76e1/lWpokVrjB9.json"
lottie_hello1 = load_lottieurl(url_lottie1)
place1 = st.empty()

with place1.container():
    anima1 , anima2 = st.columns([2,1])
    with anima1:
        # st.image("aai_black.png", width = 350, use_column_width=True)
        st.image("aai_white.png", width = 350, use_column_width=False)
    with anima2:
        st_lottie(
        lottie_hello1,
        speed=1,
        reverse=False,
        loop=True,
        quality="high", # medium ; high
        height=250,
        width=250,
        key=None,
        )

def extract_and_save_audio(video_URL, destination, final_filename):
  video = YouTube(video_URL)#get video
  audio = video.streams.filter(only_audio=True).first()#seperate audio
  output = audio.download(output_path = destination)#download and save for transcription
  _, ext = os.path.splitext(output)
  new_file = final_filename + '.mp3'
  os.rename(output, new_file)

def chunk_clips(transcription, clip_size):
  texts = []
  sources = []
  for i in range(0,len(transcription),clip_size):
    clip_df = transcription.iloc[i:i+clip_size,:]
    text = " ".join(clip_df['text'].to_list())
    source = str(round(clip_df.iloc[0]['start']/60,2))+ " - "+str(round(clip_df.iloc[-1]['end']/60,2)) + " min"
    print(text)
    print(source)
    texts.append(text)
    sources.append(source)

  return [texts,sources]

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
    
    
#App title
st.header("Youtube Question Answering Bot")
state = st.session_state
site = st.text_input("Enter your URL here")
if st.button("Build Model"):
  if site is None:
    st.info(f"""Enter URL to Build QnA Bot""")
  elif site:
    try:
      my_bar = st.progress(0, text="Fetching the video. Please wait.")
      # Set the device
      device = "cuda" if torch.cuda.is_available() else "cpu"
      
      # Load the model
      whisper_model = whisper.load_model("base", device=device)
          
      # Video to audio
      video_URL = site
      destination = "."
      final_filename = "AlphaGPT"
      extract_and_save_audio(video_URL, destination, final_filename)

      # run the whisper model
      audio_file = "AlphaGPT.mp3"
      my_bar.progress(50, text="Transcribing the video.")
      result = whisper_model.transcribe(audio_file, fp16=False, language='English')
     
      transcription = pd.DataFrame(result['segments'])

      chunks = chunk_clips(transcription, 50)
      documents = chunks[0]
      sources = chunks[1]


      my_bar.progress(75, text="Building QnA model.")
      embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
      #vstore with metadata. Here we will store page numbers.
      vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
      #deciding model
      model_name = "gpt-3.5-turbo"
      
      retriever = vStore.as_retriever()
      retriever.search_kwargs = {'k':2}
      llm = OpenAI(model_name=model_name, openai_api_key = openai_api_key)
      model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
      my_bar.progress(100, text="Model is ready.")
      st.session_state['crawling'] = True
      st.session_state['model'] = model
      st.session_state['site'] = site

    except Exception as e:
              st.error(f"An error occurred: {e}")
              st.error('Oops, crawling resulted in an error :( Please try again with a different URL.')
     
if site and ("crawling" in state):
      st.header("Ask your data")
      model = st.session_state['model']
      site = st.session_state['site']
      st.video(site, format="video/mp4", start_time=0)
      user_q = st.text_input("Enter your questions here")
      if st.button("Get Response"):
        try:
          with st.spinner("Model is working on it..."):
#             st.write(model)
            result = model({"question":user_q}, return_only_outputs=True)
            st.subheader('Your response:')
            st.write(result["answer"])
            st.subheader('Sources:')
            st.write(result["sources"])
        except Exception as e:
          st.error(f"An error occurred: {e}")
          st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
