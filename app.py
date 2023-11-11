import openai
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_feedback import streamlit_feedback
import yaml
import numpy as np

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

############ functions
def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    return user_response.update({"some metadata": 123})
############

st.title("ECOGNIZE prototype")

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
with st.spinner(text="In progress..."):
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )

    name, authentication_status, username = authenticator.login("Login", "main")
    # st.write(name, authentication_status, username)

# print(authentication_status)

# write a welcome message after the user logs in
if name is not None:
    st.info(
        f"""
        Welcome to ECOGNIZE, {name.split()[0]}! 🌍

        ECOGNIZE is a prototype that uses OpenAI's GPT to answer questions about sustainability.  It is a prototype for the Junction 2023 hackathon.
        To get started, type a question in the chat box.  The AI assistant will answer your question and provide a summary of the answer.
        To learn more about ECOGNIZE, click the "About" button in the top right corner.
        """
    )

    openai.api_key = st.secrets["openai_api_key"]
    feedback = None

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
        label_visibility="collapsed",
        index=1,
    )

    # add a notification that the user picks the most sustainable option for the model if they pick "gpt-3.5-turbo"
    if st.session_state["openai_model"] == "gpt-4":
        st.sidebar.warning(
            "You have selected the largest, least sustainable model.  Please only use this model if you need an extensive answer."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = 0

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Please provide extra information",
        "on_submit": _submit_feedback,
    }

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    for n, msg in enumerate(st.session_state.messages):
        # st.chat_message(msg["role"]).write(msg["content"])

        if msg["role"] == "assistant" and n > 1:
            feedback_key = f"feedback_{int(n/2)}"

            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = 0

            feedback = streamlit_feedback(
                **feedback_kwargs,
                key=feedback_key,
            )

    if prompt := st.chat_input("What would you like to summarize?"):
        # adjust prompt to create a summary of what the user wants to know about
        # if "list" in prompt.lower():
        #     prompt2 = ""
        prompt2 = "Answer the following query and summarize it in 1-2 paragraphs:\n" + prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
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
            # feedback = streamlit_feedback(feedback_type="thumbs")
            # add title to the chart
            st.markdown("### Sustainability score over time")
            st.bar_chart(np.random.randn(30, 3))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if feedback:
        st.write(feedback)