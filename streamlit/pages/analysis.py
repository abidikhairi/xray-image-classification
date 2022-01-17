import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
	st.title('Dataset Analysis')
	
	col1, col2, col3 = st.columns(3)

	with col1:
		st.header("Trainset")
		df = pd.read_csv('/home/flursky/Work/xray-image-classification/data/train.csv')
		
		df['category'] = list(map(lambda x: str(x), range(len(df))))
		df.loc[df['label'] == 0, 'category'] = 'normal'
		df.loc[df['label'] == 1, 'category'] = 'pneumonia'

		fig, ax = plt.subplots()

		df.groupby('category').count().plot(kind='pie', y='label', ax=ax, autopct='%1.0f%%', title='Distribution des images')

		st.dataframe(df.groupby('category').count())
		st.pyplot(fig)

	with col2:
		st.header("Testset")
		df = pd.read_csv('/home/flursky/Work/xray-image-classification/data/test.csv')
		df['category'] = range(len(df))
		df.loc[df['label'] == 0, 'category'] = 'normal'
		df.loc[df['label'] == 1, 'category'] = 'pneumonia'

		fig, ax = plt.subplots()

		df.groupby('category').count().plot(kind='pie', y='label', ax=ax, autopct='%1.0f%%', title='Distribution des images')

		st.dataframe(df.groupby('category').count())
		st.pyplot(fig)


	with col3:
		st.header("Validset")
		df = pd.read_csv('/home/flursky/Work/xray-image-classification/data/val.csv')
		
		df['category'] = range(len(df))
		df.loc[df['label'] == 0, 'category'] = 'normal'
		df.loc[df['label'] == 1, 'category'] = 'pneumonia'

		fig, ax = plt.subplots()

		df.groupby('category').count().plot(kind='pie', y='label', ax=ax, autopct='%1.0f%%', title='Distribution des images')

		st.dataframe(df.groupby('category').count())
		st.pyplot(fig)

	st.header('Training Examples')

	samples = df.sample(n=8)
	rows = [st.columns(4) for _ in range(2)]
	
	for cols in rows:
		for (col, (path, _, category)) in zip(cols, samples.itertuples(index=False)):
			with col:
				st.image(path, caption=f'Label: {category}')
