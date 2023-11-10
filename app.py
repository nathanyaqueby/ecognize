
import streamlit as st
import openai
import sumy

# Set up OpenAI API authentication
openai.api_key = st.secrets["openai_api_key"]

# Set up Streamlit app
st.title("Sustainable Chatbot")
query = st.text_input("Enter your query:")
if st.button("Submit"):
    # Use GPT-4 to generate a summary of the relevant information from the internet
    summary = openai.Completion.create(
        engine="davinci",
        prompt=query,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    ).choices[0].text
    
    # Use sumy to summarize the summary into a more concise and readable format
    parser = sumy.parsers.plaintext.PlaintextParser.from_string(summary, sumy.tokenizers.Tokenizer("english"))
    summarizer = sumy.summarizers.lex_rank.LexRankSummarizer()
    summary = summarizer(parser.document, 3)
    
    # Display the summarized information to the user
    st.write(summary)
