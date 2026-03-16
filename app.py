import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import PyPDF2

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")

st.title("📚 AI Study Assistant")
st.write("Upload notes and generate summaries, key points, and quiz questions.")

# pipeline
summarizer = pipeline("text-generation", model="google/flan-t5-base")

uploaded_file = st.file_uploader("Upload your notes (PDF or TXT)")

text = ""

if uploaded_file is not None:

    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()

    else:
        text = uploaded_file.read().decode("utf-8")

    st.subheader("Preview of Notes")
    st.write(text[:1000])

if st.button("Generate AI Summary"):

    if text == "":
        st.warning("Please upload a file first.")
    else:

        prompt = f"Summarize the following notes and provide key points:\n{text[:1500]}"

        summary = summarizer(prompt, max_length=200)
        result = summary[0]['generated_text']

        st.subheader("AI Generated Summary")
        st.write(result)

        # DOWNLOAD FEATURE
        st.download_button(
            label="Download Summary",
            data=result,
            file_name="summary.txt",
            mime="text/plain"
        )

        # VISUAL ANALYTICS
        st.subheader("Text Analytics")

        words = text.lower().split()

        stopwords = ["the","is","and","in","to","of","a","for","on","with"]

        filtered_words = [word for word in words if word not in stopwords]

        word_counts = Counter(filtered_words)

        common_words = word_counts.most_common(10)

        labels = [word[0] for word in common_words]
        values = [word[1] for word in common_words]

        fig, ax = plt.subplots()
        ax.bar(labels, values)

        ax.set_title("Top Keywords in Notes")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
