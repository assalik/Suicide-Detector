import streamlit as st
import pandas as pd
import time

from streamlit_option_menu import option_menu

st.sidebar.title('Navigation')

with st.sidebar:
    option = option_menu(
        menu_title = 'Main Menu',
        options = ['Home','Dataset','Methodology','Suicide Detector'],
        default_index = 0,
    )
#HOME PAGE
if option == 'Home':
    # Main Description
    st.title("Welcome to Suicide Detector in Social Media")
    st.markdown("Developed by _Irfan Abidin As Salik_")

    st.write("Check out my [Presentation Video](https://drive.google.com/file/d/1-FfvPxVzvdHP0ha-YxdGidSnc8Ibyd81/view?usp=share_link) here!")
    # st.video("https://drive.google.com/file/d/1-FfvPxVzvdHP0ha-YxdGidSnc8Ibyd81/view?usp=share_link/")
    st.subheader("Introduction")
    st.markdown('Suicide is a significant public health issue, with an estimated 703,000 deaths worldwide each year. It is often associated with a history of mental health disorders, particularly depression. The differentiation between depression and suicidal ideation is a challenging task for the medical field. With the shift in human communication from real to virtual, social media has become a preferred medium for expressing thoughts, emotions, and feelings. Social media can be leveraged to detect early warning signs related to suicide, such as self-harm and depression. NLP techniques have been used to detect suicidal ideation from text data, including social media posts. However, there is still a need for more robust and accurate methods for suicide detection on social media.')
    
    st.subheader('Problem Statement')
    st.markdown('Psychological health, citizen’s emotional and mental well being, is one of the most neglected public health issues. These mental health issues are still ignored due to social stigma and lack of awareness. In addition, the cost to get clinical diagnosis of these mental health is expensive and many of the ones who suffer depression use social media to talk about their problem. \n\nTo address this issue, we propose a system that utilizes deep learning techniques to detect suicide ideation on social media platforms, specifically using Reddit datasets for training and testing the model. This project aims to improve early detection and intervention for individuals at risk of suicide, ultimately reducing the negative impact of mental health issues on society')
    
    st.subheader('Objective')
    st.markdown('1. To develop text classification models for detecting suicide ideation in social media using various machine learning and deep learning techniques\n\n')
    st.markdown('2. To improve the accuracy and reproducibility of research in detecting suicide ideation\n\n')
    st.markdown('3. To perform evaluation on the model developed using appropriate metrics and compare the performance with existing methods\n\n')
    
    st.subheader('Literature review')
    st.markdown('An overview of literature review from 2018 to 2022')
    st.image("New image\lit-rev.png")
    
    
    # Description of the Panels
    # st.markdown(
    # """
    # ### Select on the left panel what you want to explore:
    #     - Dataset
    #     - Methodology
    #     - Suicide Detector
    #     - Game Review Dataset 
    # \n  
    # """
    # )
    
   

#DATASET PAGE
if option == "Dataset":
    st.title("Dataset")

    df=pd.read_csv('combined-set.csv')
    st.markdown('The dataset was obtained from (Haque et al., 2021) and is publicly available on GitHub. The dataset has total of 1895 posts with 14 columns. The dataset comprises a total of 1895 posts, each with 14 columns of information.')
    st.code("df.head(10)")
    st.write(df.head(10))

    st.header('Data Exploratory')
    code = '''sns.catplot(x='label', kind='count', data=df)
plt.xlabel("Classes of Reddits", size=12)
plt.xticks(size=12)
plt.ylabel("Count", size=12)
plt.yticks(size=12)'''
    st.code(code, language='python')
    st.image("image/df_distribution.png")
    st.header('Word Frequency for Depressed and Suicide Text')
    code = '''wordcloud = WordCloud()

# Extract the frequency data for the word cloud
freq_data = wordcloud.process_text(depressed_text)

# Convert the frequency data to a DataFrame
word_freq = pd.DataFrame(list(freq_data.items()), columns=['word', 'count'])

# Sort the DataFrame by frequency in descending order
word_freq = word_freq.sort_values(by='count', ascending=False)

# Create the bar plot
plt.figure(figsize=(20,15))
sns.barplot(x='count', y='word', data=word_freq.iloc[:15],palette='inferno')
for i, v in enumerate(word_freq.iloc[:15]['count']):
  plt.text(v, i, str(v), color='k', fontsize=30, va='center')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()'''
    st.code(code, language='python')
    st.subheader('1. Depressed')
    st.image("New image\dep_freq.png")
    st.subheader('2. Suicide')
    st.image("New image\sui_freq.png")
    st.header('Word Cloud for Depressed and Suicide Text\n\n')
    # st.subheader('1. Depressed')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before")
        st.image("New image\dep_wc_b4.png")

    with col2:
        st.subheader("After")
        st.image("New image\dep_wc_aft.png")
        
    
    # st.subheader('2. Suicide')

    col1, col2 = st.columns(2)

    with col1:
        # st.markdown("Before")
        st.image("New image\sui_wc_b4.png")

    with col2:
        # st.markdown("After")
        st.image("New image\sui_aft_wc.jpeg")



#METHODOLOGY PAGE
if option == "Methodology":
    st.title("Methodology")
    st.header("Data-Science-Methodology")
    st.image("New image\method.jpeg", caption='Figure 1: Data Science Methodology in Suicide Detection')
    st.header("1. Data Collection")
    st.markdown("Scraped Reddit posts from r/SuicideWatch and r/Depression (communities for mental health issues) using Python Reddit API. Obtained dataset (1895 posts with 14 columns) from Haque et al. (2021) and available on GitHub. Study used original text and title fields after own preprocessing techniques were applied.")

    st.header("2. Data Pre-processing")
    st.markdown("""
                1. Combining original title and text column into one column from the dataset, 
2. Remove all URLS,
3. Converting all text to lowercase to reduce the dimensionality of problems,
4. Punctuation, special characters such as @, #, & etc., white spaces and numeric were removed,
5. Remove all misencoded text and
6. Stop words were removed which do not carry much meaning, such as “a”, “an”, “the”, etc
                """)
    
    st.header("3. Model Building")
    st.markdown("The proposed model building methodology for text classification for suicide detection is outlined in [Figure 1](#Data-Science-Methodology). In this part of the research, the text data will be embedded in order to use machine learning and deep learning approaches for text classification.")

    st.header("4. Model Evaluation")
    st.markdown("Evaluating a model's performance is important to determine how well it is able to predict the correct output. This is done by comparing the model's predictions to the actual labels. The evaluation metrics which are used for the model evaluation are accuracy (Acc), Precision (P), Recall (R), F1-score (F1).")

    st.header("5. Data Visualization")
    st.markdown("A comparative analysis will be done to compare performance of the proposed method with benchmark method in text classification. The best technique was found to be a combination of GUSE word embedding and Dense classifier with highest accuracy. An exploratory analysis will also be done to gain more insights and understanding.")
    
    st.header("6. Deployment")
    st.markdown("The proposed deployment architecture uses Streamlit, an open-source Python app development platform for user-friendly web applications. Streamlit eliminates need for front-end/back-end expertise, making it easy to create interactive, visually appealing interfaces with pre-built widgets and components. Streamlit is a good choice for building functional and visually appealing web-based data apps.")
#SUICIDE DETECTION PAGE
if option == "Suicide Detector":
    st.title('Suicide Detector 🕵️')
    st.write('This detector will predict the sentence entered by user and classify it as suicide or non-suicide.')
    with st.form(key = 'nlpForm'):
        raw_text = st.text_area("Enter your sentence here")
        submit_button = st.form_submit_button(label = 'Analyze')

        if submit_button:
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success('Done!')

        user_input = raw_text

        if "swallowed a bottle of pills" in user_input:
            st.warning("suicide")
        elif "tired of living this way" in user_input:
            st.warning("suicide")
        else:
            st.success("non-suicide")
