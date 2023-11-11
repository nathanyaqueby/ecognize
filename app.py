import openai
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_feedback import streamlit_feedback
import yaml
import numpy as np
import pandas as pd
from streamlit_extras.app_logo import add_logo
from trubrics.integrations.streamlit import FeedbackCollector
import uuid


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

with open("style.css", "r") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

add_logo("ecognize logo.png", height=100)

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

# put logo in the center
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")
with col2:
    st.image("ecognize logo.png", use_column_width=True)
with col3:
    st.write("")

collector = FeedbackCollector(
    project="ecognize",
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
)

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
if authentication_status:
    user_points = load_user_points(username)
    st.sidebar.title(f"Hello, {name.split()[0]}!")

    # compute the average points of all users
    user_points_pd = pd.read_csv("user_points.csv")
    average_points = user_points_pd["user_points"].mean()

    # add a st.metric to show how much the user's points are above or less than the average in percentage
    if user_points > average_points:
        st.sidebar.metric("Your points", f"{user_points}", f"{round((user_points - average_points) / average_points * 100, 2)} %")
    elif user_points < average_points:
        st.sidebar.metric("Your points", f"{user_points}", f"-{round((user_points - average_points) / average_points * 100, 2)} %")
    else:
        st.sidebar.metric("Your points", f"{user_points}", f"Average", delta_color="off")

    # st.sidebar.markdown(f"""
    #                     <p style='font-family': Garet'>Hello, {name.split()[0]}! <p> <br>
    #                     <p style='font-family': Garet'>Your points: {user_points}</p>
    #                     """, unsafe_allow_html=True)
                        

    # rewrite st info with html font family Garet
    st.markdown("""
                <p style='font-family': Garet'>Welcome to <b>PROMPTERRA</b> by <b>ECOGNIZE</b> 🌍</p> <br>
                <p style='font-family': Garet'>PROMPTERRA is a platform that trains users to use OpenAI's GPT in a more sustainable way. To get started, type a prompt in the chat box on the left and click enter. The AI will respond with a summary of your prompt. You can then provide feedback on the response to gain points!</p>
                """, unsafe_allow_html=True)

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
    if "prompt_ids" not in st.session_state:
        st.session_state["prompt_ids"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Please provide extra information",
        "on_submit": _submit_feedback,
    }

    # Assign IDs to existing messages if they don't have one
    for n, msg in enumerate(st.session_state["messages"]):
        st.chat_message(msg["role"]).write(msg["content"])

        if msg["role"] == "assistant":
            if n > 0:
                feedback_key = f"feedback_{int(n / 2)}"

                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None
                feedback = collector.st_feedback(
                    component="default",
                    feedback_type="thumbs",
                    open_feedback_label="[Optional] Provide additional feedback",
                    model=st.session_state["openai_model"],
                    key=feedback_key,
                    prompt_id=st.session_state.prompt_ids[int(n / 2) - 1],
                    user_id=st.secrets["TRUBRICS_EMAIL"]
                )
            else:
                feedback_key = f"feedback_{n}"

                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None
                feedback = collector.st_feedback(
                    component="default",
                    feedback_type="thumbs",
                    open_feedback_label="[Optional] Provide additional feedback",
                    model=st.session_state["openai_model"],
                    key=feedback_key,
                    prompt_id=st.session_state.prompt_ids[n],
                    user_id=st.secrets["TRUBRICS_EMAIL"]
                )

            if feedback:
                st.session_state['feedback'][feedback_key] = feedback
                # Assuming 1 point for each feedback
                user_points = update_user_points(username, 2)
                # add a notification that the user has earned a point
                st.sidebar.success(
                    f"You have earned a point!  Your points: {user_points}"
                )

    if prompt := st.chat_input("What would you like to summarize?"):
        # adjust prompt to add sources in a specified format
        prompt2 = "Answer the following query and summarize it in 1-2 paragraphs:\n" + prompt + " Write the sources you used in the following format: - Source 1: [link] - Source 2: [link] - Source 3: [link]"

        # prompt2 = "Answer the following query and summarize it in 1-2 paragraphs:\n" + prompt
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
                temperature=0.2,
                max_tokens=300,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")

            logged_prompt = collector.log_prompt(
                config_model={"model": st.session_state["openai_model"]},
                prompt=prompt,
                generation=full_response,
                session_id=st.session_state.session_id,
                tags="prompterra",
                user_id=str(st.secrets["TRUBRICS_EMAIL"])
                )
            st.session_state.prompt_ids.append(logged_prompt.id)

            message_placeholder.markdown(full_response)
            # add title to the chart
            # st.markdown("### Sustainability score over time")
            # st.bar_chart(np.random.randn(30, 3))

        # After getting the response, add it to the session state
        st.session_state['messages'].append({"role": "assistant", "content": full_response, "id": new_message_id + 1})

elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")

elif st.session_state["authentication_status"] is None:
    st.info("Please enter your username and password")
    # if feedback:
    #     st.write(feedback)