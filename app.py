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
from pymongo import MongoClient
from bson import ObjectId
from search import search_bing
import json


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

# Replace the following with your own connection string
MONGO_URI = st.secrets["MONGO_URI"]
# Connect to the MongoDB Atlas cluster
client = MongoClient(MONGO_URI)

# Select your database
db = client["junction"]

# Select your collection
collection = db["sustainability_scores"]

def update_user(username, user_point, sustainability_score):
    """ Update the user's points and sustainability score based on username """
    user_document = collection.find_one({"username": username})
    if user_document:
        result = collection.update_one(
            {"_id": user_document["_id"]},
            {"$set": {"user_point": user_point, "sustainability_score": sustainability_score}}
        )
        if result.matched_count > 0:
            print(f"User {username} updated. Points: {user_point}, Sustainability Score: {sustainability_score}")
        else:
            print(f"Update operation did not find the user {username}")
    else:
        print(f"No user found with the username {username}")

def load_from_mongo(username):
    """ Fetch a single document based on username """
    query = {"username": username}
    document = collection.find_one(query)
    return document
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
    # user_points = load_user_points(username)
    st.sidebar.title(f"Hello, {name.split()[0]}!")

    # compute the average points of all users
    user_pd = load_from_mongo(username)
    user_points = user_pd["user_point"]
    user_num_query = user_pd["sustainability_score"]

    # load all users nqueby, tianyi, cmakafui, angelineov, outokumpu from mongo
    users = ["nqueby", "tianyi", "cmakafui", "angelineov", "outokumpu"]
    sustainability_scores = []
    user_points = 0

    for user in users:
        user_document = load_from_mongo(user)
        sustainability_scores.append(user_document["sustainability_score"])
        user_points += user_document["user_point"]

    average_points = user_points / len(users)

    # get the user's number of query from the database
    average_query = np.mean(sustainability_scores)

    # create two cols
    col41, col42 = st.sidebar.columns(2)

    with col41:
        # add a st.metric to show how much the user's points are above or less than the average in percentage
        if user_points > average_points:
            st.metric("Your points", f"{user_points} 🌍", f"{round((user_points - average_points) / average_points * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        elif user_points < average_points:
            st.metric("Your points", f"{user_points} 🌍", f"-{round((user_points - average_points) / average_points * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        else:
            st.metric("Your points", f"{user_points} 🌍", f"Average", delta_color="off", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
    
    with col42:
        # add a st.metric to show the user's number of query
        if user_num_query > average_query:
            st.metric("Eco-friendly queries", f"{user_num_query} 🌿", f"{round((user_num_query - average_query) / average_query * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        elif user_num_query < average_query:
            st.metric("Eco-friendly queries", f"{user_num_query} 🌿", f"-{round((user_num_query - average_query) / average_query * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        else:
            st.metric("Eco-friendly queries", f"{user_num_query} 🌿", f"Average", delta_color="off", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")


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
    # st.sidebar.title("Leaderboard")
    # load the csv file with the user points
    # sort the users by points
    # user_pd = user_pd.sort_values(by=["user_points"], ascending=False)
    # if len(user_pd) >= 5:
    #     # Create a new DataFrame for top 5 users
    #     top_users_data = {
    #         "Rank": ["🥇", "🥈", "🥉", "🏅", "🏅"],
    #         "Username": [user_pd.iloc[i]['username'] for i in range(5)],
    #         "Points": [user_pd.iloc[i]['user_points'] for i in range(5)]
    #     }

        # top_users_df = pd.DataFrame(top_users_data)
        # st.sidebar.dataframe(top_users_df, hide_index=True, use_container_width=True)

    # should be the end of the sidebar
    with st.sidebar:
        authenticator.logout("Logout", "main", key="unique_key")

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
                update_user(username, 2, 0)
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
            # For streaming, we need to loop through the response generator
            reply_text = ""
            function_name = ""
            function_args = ""
            for chunk in openai.ChatCompletion.create(
                model="gpt-35-turbo",
                deployment_id="gpt-35-turbo",
                messages=st.session_state['messages'],
                max_tokens=300,
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

elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")

elif st.session_state["authentication_status"] is None:
    st.info("Please enter your username and password")
    # if feedback:
    #     st.write(feedback)