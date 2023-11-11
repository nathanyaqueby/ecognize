import openai
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_feedback import streamlit_feedback
import yaml
import numpy as np
import pandas as pd

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

def load_user_points(username):
    if user_points := st.session_state.get('user_points', {}).get(username, None):
        return user_points
    # load the csv file with the user points
    user_points_pd = pd.read_csv("user_points.csv")
    # get the points for the user
    points = user_points_pd[user_points_pd["username"] == username]["user_points"].values[0]
    return points

def update_user_points(username, points):
    # Load the current points from the CSV
    user_points_pd = pd.read_csv("user_points.csv")

    # Check if the user exists in the DataFrame
    if username in user_points_pd['username'].values:
        # Update the user's points
        user_points_pd.loc[user_points_pd['username'] == username, 'user_points'] += points
    else:
        # Add new user to the DataFrame
        new_row = {'username': username, 'user_points': points}
        user_points_pd = user_points_pd.append(new_row, ignore_index=True)

    # Save updated DataFrame to CSV
    user_points_pd.to_csv("user_points.csv", index=False)

    # Update session state
    new_points = user_points_pd[user_points_pd['username'] == username]['user_points'].values[0]
    st.session_state.setdefault('user_points', {})[username] = new_points
    return new_points
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

# write a welcome message after the user logs in
if name is not None:
    user_points = load_user_points(username)
    st.sidebar.write(f"Hello, {name.split()[0]}! Your points: {user_points}")

    st.info(
        f"""
        Welcome to **ECOGNIZE** 🌍

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

    # show a ranking of the user points
    st.sidebar.title("Leaderboard")
    # load the csv file with the user points
    user_points_pd = pd.read_csv("user_points.csv")
    # sort the users by points
    user_points_pd = user_points_pd.sort_values(by=["user_points"], ascending=False)
    if len(user_points_pd) >= 5:
        # Create a new DataFrame for top 5 users
        top_users_data = {
            "Rank": ["🥇", "🥈", "🥉", "🏅", "🏅"],
            "Username": [user_points_pd.iloc[i]['username'] for i in range(5)],
            "Points": [user_points_pd.iloc[i]['user_points'] for i in range(5)]
        }

        top_users_df = pd.DataFrame(top_users_data)
        st.sidebar.dataframe(top_users_df, hide_index=True, use_container_width=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Please provide extra information",
        "on_submit": _submit_feedback,
    }

    # Assign IDs to existing messages if they don't have one
    for i, message in enumerate(st.session_state.get('messages', [])):
        if 'id' not in message:
            message['id'] = i

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        # Check if feedback should be displayed for this message
        message_id = f"feedback_{message['id']}"
        if message['role'] == 'assistant' and message_id not in st.session_state['feedback']:
            feedback = streamlit_feedback(
                **feedback_kwargs,
                key=message_id
            )
            if feedback:
                st.session_state['feedback'][message_id] = feedback
                # Assuming 1 point for each feedback
                user_points = update_user_points(username, 1)
                # add a notification that the user has earned a point
                st.sidebar.success(
                    f"You have earned a point!  Your points: {user_points}"
                )

    if prompt := st.chat_input("What would you like to summarize?"):
        # adjust prompt to create a summary of what the user wants to know about
        # if "list" in prompt.lower():
        #     prompt2 = ""
        prompt2 = "Answer the following query and summarize it in 1-2 paragraphs:\n" + prompt
        new_message_id = len(st.session_state['messages'])  # Unique ID for the new message
        st.session_state['messages'].append({"role": "user", "content": prompt, "id": new_message_id})

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
            # add title to the chart
            st.markdown("### Sustainability score over time")
            st.bar_chart(np.random.randn(30, 3))

        # After getting the response, add it to the session state
        st.session_state['messages'].append({"role": "assistant", "content": full_response, "id": new_message_id + 1})

    if feedback:
        st.write(feedback)