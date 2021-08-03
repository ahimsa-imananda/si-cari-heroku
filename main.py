import pandas as pd
import streamlit as st
import time
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

def preprocessing(df, tokenizer):
	# To Lower Case
    df = to_lower_case(df)
    # Stemming and Remove Stopwords
    #df = stemming_remove_stopwords(df)
    #tokenize pad sequence
    X = tokenize_pad_sequence(tokenizer, df)
    return X

def to_lower_case(df):
	df["title_text"] = [entry.lower() for entry in df["title_text"]]
	return df

def tokenize_pad_sequence(tokenizer, df):
	X = df["title_text"].values
	X = tokenizer.texts_to_sequences(X)
	max_length = 250
	X = pad_sequences(X, maxlen=max_length)
	return X

def load_stuff():
	loaded_model = tf.keras.models.load_model("resources/model_lstm_glove.h5")
	with open ('resources/tokenizer.pickle', 'rb') as f:
		temp_tokenizer = pickle.load(f)
	return loaded_model, temp_tokenizer


st.title("SI-CARI - Cek Berita Asli atau Palsu!")
st.write('')
st.write("Periksa keaslian suatu berita yang Anda temukan di internet sebelum terperdaya!")
st.write("Website ini dapat memprediksi berita hoax menggunakan Deep Learning.")
st.subheader("Lengkapi isian dibawah untuk melakukan pemeriksaan Berita!")
st.write('Isian hanya dapat diisi menggunakan teks berita Bahasa Inggris.')
st.write('Hal ini dikarenakan, model deep learning dilatih menggunakan dataset teks berita Bahasa Inggris.')

# Add a selectbox to the sidebar:
st.sidebar.header("Terima Kasih Sudah Menggunakan SI-CARI!")
st.sidebar.write("Website ini adalah sebuah prototipe yang mengimplementasikan model klasifikasi berita palsu menggunakan Machine Learning")
st.sidebar.write("Model klasifikasi dilatih menggunakan dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
st.sidebar.write("Semoga hasil dari Website ini membantu Anda mempertimbangkan keaslian suatu berita.")
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.subheader("Contact me at: ahimsa.imananda@gmail.com")

title_news = st.text_input("Judul berita:")
st.text('')
text_news = st.text_area("Isi Teks Berita:")

title_news_translated, text_news_translated = translate_input(title_news, text_news)
data_teks = {'title_text': title_news_translated + " " + text_news_translated}
df = (pd.DataFrame(data_teks, index=[0]))
model , tokenizer = load_stuff()
if st.button('Submit'):
	with st.spinner('SI-CARI Sedang memeriksa masukkan pengguna. Mohon tunggu sebentar....'):
		my_bar = st.progress(0)
		for percent_complete in range(100):
			time.sleep(0.02)
			my_bar.progress(percent_complete + 1)	
	st.header('Hasil pemeriksaan berita:')
	X = preprocessing(df, tokenizer)
	predictions = model.predict(X)
	if predictions[0][0] > 0.95:
		confident = predictions[0][0] * 100
		confident = int(confident)
		st.error('Berita PALSU')
		st.write('Confident: ' +  str(confident) + " %")
	elif predictions[0][0] < 0.05:
		confident = 100 - predictions[0][0] * 100
		confident = int(confident)
		st.success('Berita ASLI')
		st.write('Confident: ' + str(confident) + " %")
	else:
		confident = predictions[0][0] * 100
		st.warning("SI-CARI tidak yakin ini Berita ASLI atau PALSU")
		st.write('Confident: ' + str(confident) + " %")
