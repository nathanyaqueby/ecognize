import openai
import streamlit as st
import json
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_feedback import streamlit_feedback
import yaml
import numpy as np
import pandas as pd
from streamlit_extras.app_logo import add_logo
from trubrics.integrations.streamlit import FeedbackCollector
from search import search_bing
import uuid

TIMEOUT = 60

RETRIEVAL_PROMPT = """
You are a powerful AI chat assistant that can answer user questions by retrieving relevant information from various sources. If you call any functions, please follow strictly the function descriptions and infer the parameters from the predefined ones based on the message history until this point. Do not make up your own function call parameters that are not defined.
"""

GENERATION_PROMPT = """
You are a powerful AI chat assistant that can answer user questions by retrieving relevant information from various sources. Be careful to answer the question using only the information from function calls. If they do not return any answers or the answers don't match the question, just say you cannot answer the question and stop there. 
If you used one or several retrieved information sources in your answer, please cite the relevant sources at the end of your response, starting with the text "SOURCES: " (always in plural), followed by a JSON standard formatted structure, as in below sample: 
SOURCES: {
    "sources": [
        "source/url 1", 
        "source/url 2 etc"
    ]
}
"""

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

############ Function calling
functions = [
    {
        "name": "search_bing",
        "description": "Search for relevant hits from Bing.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Query string to search the posts for, inferred from the user message and ALWAYS translated to ENGLISH before passing on to the function. Sample query styles: 'expected development of stainless steel market pricing in December 2022' or 'possible energy price developments over January-March 2023'. Note: today is {dt.datetime.utcfromtimestamp(UTC_TIMESTAMP).strftime('%Y-%m-%d')}."
                },
                "search_index": {
                    "type": "string",
                    "description": "Name of the Bing Search index to use. Valid choices: 'DEFAULT'. If 'DEFAULT' is used, the search will be performed on the entire web."
                }
            },
            "required": ["query", "search_index"]
        }
    }
]

available_functions = {
    "search_bing": search_bing,
}

st.title("PROMPTERRA")

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
if name is not None:
    user_points = load_user_points(username)
    st.sidebar.write(f"Hello, {name.split()[0]}! Your points: {user_points}")

    st.info(
        f"""
        Welcome to **PR🌍MPTERRA** by EC🌍GNIZE 

        PR🌍MPTERRA is a platform that trains users to use OpenAI's GPT in a more sustainable way.
        """
    )

    feedback = None

    # # create a dropdown to select the model
    # st.sidebar.title("Model")
    # st.sidebar.markdown(
    #     "Select the model you want to use. The turbo model is faster but less accurate."
    # )
    # # dropdown to select the model
    # st.session_state["openai_model"] = st.sidebar.selectbox(
    #     "Model",
    #     [
    #         "gpt-3.5-turbo",
    #         "gpt-4",
    #         "gpt-4-1106-preview"
    #     ],
    #     label_visibility="collapsed",
    #     index=1,
    # )

    # # add a notification that the user picks the most sustainable option for the model if they pick "gpt-3.5-turbo"
    # if st.session_state["openai_model"] == "gpt-4":
    #     st.sidebar.warning(
    #         "You have selected the largest, least sustainable model.  Please only use this model if you need an extensive answer."
    #     )

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
        st.session_state.messages = [{"role": "system", "content": RETRIEVAL_PROMPT}]
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
                    model="gpt-4",
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
                    model="gpt-4",
                    key=feedback_key,
                    prompt_id=st.session_state.prompt_ids[n],
                    user_id=st.secrets["TRUBRICS_EMAIL"]
                )

            if feedback:
                st.session_state['feedback'][feedback_key] = feedback
                # Assuming 1 point for each feedback
                user_points = update_user_points(username, 1)
                # add a notification that the user has earned a point
                st.sidebar.success(
                    f"You have earned a point! Your points: {user_points}"
                )

    if prompt := st.chat_input("Ask me anything"):

        # Add the user message to the session state and render it
        st.session_state['messages'].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # For streaming, we need to loop through the response generator
            reply_text = ""
            function_name = ""
            function_args = ""
            for chunk in openai.ChatCompletion.create(
                model="gpt-35-turbo",
                deployment_id="gpt-35-turbo",
                messages=st.session_state['messages'],
                max_tokens=200,
                timeout=TIMEOUT,
                function_call="auto",
                functions=functions,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", None)
                function_call = delta.get("function_call", None)
                if function_call is not None:
                    function_name += function_call.get("name", "")
                    function_args += function_call.get("arguments", "")
                if content is not None:
                    reply_text += content

                    # Sanitize output
                    if reply_text.startswith("AI: "):
                        reply_text = reply_text.split("AI: ", 1)[1]
                
                # Continuously write the response in Streamlit
                message_placeholder.markdown(reply_text)

            # Collect full function call
            if function_name != "" and function_args != "":
                function_call = {"name": function_name, "arguments": function_args}
            else:
                function_call = None

            if function_call is None:  # Not a function call, return normal message
                # Sanitize
                if reply_text.startswith("AI: "):
                    reply_text = reply_text.split("AI: ", 1)[1]

                message_placeholder.markdown(reply_text)

            # Model wants to call a function and call it, if appropriate
            else:
                # Read the function call from model response and execute it
                fun_name = function_call.get("name", None)
                if fun_name is not None and fun_name and fun_name in available_functions:
                    function = available_functions[fun_name]
                else:
                    function = None
                fun_args = function_call.get("arguments", None)
                if fun_args is not None and isinstance(fun_args, str):
                    fun_args = json.loads(fun_args)

                query = fun_args.get("query", None)

                # Check in the cache if the response is already there and if so just return the relevant answer
                # NOT IMPLEMENTED YET - GO STRAIGHT TO RETRIEVAL AND NEW GENERATION

                if function is None:
                    fun_res = ["Error, no function specified"]
                elif query is None:
                    fun_res = ["Error, no query specified"]
                else:
                    with st.status(f"Called function `{fun_name}`"):
                        st.json(fun_args, expanded=True)
                        fun_res = function(fun_args)

                # Build an abridged, temporary message to be fed into a more powerful GPT-4 model to limit the number of tokens

                # Proceed to the secondary call to generate results
                messages = [{"role": "system", "content": GENERATION_PROMPT}]
                if query is not None:
                    messages.append({"role": "user", "content": query})
                messages.extend([{"role": "function", "name": fun_name, "content": one_fun_res} for one_fun_res in fun_res])

                # For streaming, we need to loop through the response generator
                reply_text = ""
                for chunk in openai.ChatCompletion.create(
                    model="gpt-4",
                    deployment_id="gpt-4",
                    messages=messages,
                    max_tokens=1500,
                    timeout=TIMEOUT,
                    function_call=None,
                    functions=None,
                    stream=True,
                ):
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", None)
                    function_call = delta.get("function_call", None)
                    if function_call is not None:
                        function_name += function_call.get("name", "")
                        function_args += function_call.get("arguments", "")
                    if content is not None:
                        reply_text += content

                        # Sanitize output
                        if reply_text.startswith("AI: "):
                            reply_text = reply_text.split("AI: ", 1)[1]
                    
                    # Continuously write the response in Streamlit
                    message_placeholder.markdown(reply_text)

                # # Collect full function call
                # if function_name != "" and function_args != "":
                #     function_call = {"name": function_name, "arguments": function_args}
                # else:
                #     function_call = None
                    
                # Sanitize
                if reply_text.startswith("AI: "):
                    reply_text = reply_text.split("AI: ", 1)[1]

                message_placeholder.markdown(reply_text)

                logged_prompt = collector.log_prompt(
                    config_model={"model": "gpt-4"},
                    prompt=prompt,
                    generation=reply_text,
                    session_id=st.session_state.session_id,
                    # tags=tags,
                    user_id=str(st.secrets["TRUBRICS_EMAIL"])
                    )
                
                st.session_state.prompt_ids.append(logged_prompt.id)

        # After getting the response, add it to the session state
        st.session_state['messages'].append({"role": "assistant", "content": reply_text})

    # if feedback:
    #     st.write(feedback)