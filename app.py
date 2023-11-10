import openai
import streamlit as st

st.set_page_config(
    layout="wide",
    # initial_sidebar_state="expanded",
    page_title="ECOGNIZE",
    page_icon="🌍",
    menu_items={
        "Get Help": "https://www.github.com/nathanyaqueby/ecognize/",
        "Report a bug": "https://www.github.com/nathanyaqueby/ecognize/issues",
        "About": "Unlike OpenAI, our default model is most sustainable one. Learn to adjust your prompts to help the planet!",
    },
)

st.title("ECOGNIZE prototype")

openai.api_key = st.secrets["openai_api_key"]

# create a dropdown to select the model
st.sidebar.title("Model")
st.sidebar.markdown(
    "Select the model you want to use. The turbo model is faster but less accurate."
)
# dropdown to select the model
st.session_state["openai_model"] = st.sidebar.selectbox(
    "Model",
    [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-1106-preview"
    ],
    index=1,
)

# add a notification that the user picks the most sustainable option for the model if they pick "gpt-3.5-turbo"
if st.session_state["openai_model"] == "gpt-4":
    st.sidebar.warning(
        "You have selected the turbo model. This model is faster but less accurate. Please only use this model if you need a quick answer."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to summarize?"):
    # adjust prompt to create a summary of what the user wants to know about
    prompt2 = "Answer the following query and summarize it in 1-2 paragraphs:\n" + prompt
    st.session_state.messages.append({"role": "user", "content": prompt2})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=200,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})