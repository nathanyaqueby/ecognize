import openai
import streamlit as st
import json
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_feedback import streamlit_feedback
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from streamlit_extras.app_logo import add_logo
from trubrics.integrations.streamlit import FeedbackCollector
from search import search_bing
import uuid
import datetime as dt
from langchain.embeddings.openai import OpenAIEmbeddings

UTC_TIMESTAMP = int(dt.datetime.utcnow().timestamp())
from pymongo import MongoClient
from bson import ObjectId
import re

TIMEOUT = 60
CACHE_SIMILARITY_THRESHOLD = 0.92   # Found by experimentation

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

# Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(
    layout="wide",
    # initial_sidebar_state="expanded",
    page_title="PR🌍MPTERRA by ECOGNIZE",
    page_icon="🌍",
    menu_items={
        "Get Help": "https://www.github.com/nathanyaqueby/ecognize/",
        "Report a bug": "https://www.github.com/nathanyaqueby/ecognize/issues",
        "About": "Learn to adjust your calls to help the planet!",
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

    new_user_point = user_document["user_point"] + user_point
    new_sustainability_score = user_document["sustainability_score"] + sustainability_score

    if user_document:
        result = collection.update_one(
            {"_id": user_document["_id"]},
            {"$set": {"user_point": new_user_point, "sustainability_score": new_sustainability_score}}
        )
        if result.matched_count > 0:
            print(f"User {username} updated. Points: {new_user_point}, Sustainability Score: {new_sustainability_score}")
        else:
            print(f"Update operation did not find the user {username}")
    else:
        print(f"No user found with the username {username}")

def initialize_cache():
    # LOCAL VERSION: load "cache.csv" file into pandas DataFrame if it exists, otherwise create a new one
    if Path("cache.csv").is_file():
        return pd.read_csv("cache.csv")
    else:
        return pd.DataFrame(columns=["query", "embedding", "answer", "expires_at"])
    
def add_to_cache(query, embedding, answer, expires_at):
    pass


# Callback function for refresh button
def refresh_metrics():
    average_points, average_query, user_points, user_num_query = load_all_from_mongo(username)
    st.session_state['metrics'] = (average_points, average_query, user_points, user_num_query)

def load_from_mongo(username):
    """ Fetch a single document based on username """
    query = {"username": username}
    document = collection.find_one(query)
    return document

def load_all_from_mongo(username):
    # compute the average points of all users
    user_pd = load_from_mongo(username)
    user_points = user_pd["user_point"]
    user_num_query = user_pd["sustainability_score"]

    # load all users nqueby, tianyi, cmakafui, angelineov, outokumpu from mongo
    users = ["nqueby", "tianyi", "cmakafui", "angelineov", "outokumpu"]
    sustainability_scores = []
    user_total_points = 0

    for user in users:
        user_document = load_from_mongo(user)
        sustainability_scores.append(user_document["sustainability_score"])
        user_total_points += user_document["user_point"]

    average_points = user_total_points / len(users)

    # get the user's number of query from the database
    average_query = np.mean(sustainability_scores)

    return average_points, average_query, user_points, user_num_query

def add_metrics(cola, colb, username):

    average_points, average_query, user_points, user_num_query = st.session_state['metrics']

    with cola:
        # add a st.metric to show how much the user's points are above or less than the average in percentage
        if user_points > average_points:
            st.metric("Your points", f"{user_points} 🌍", f"{round((user_points - average_points) / average_points * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        elif user_points < average_points:
            st.metric("Your points", f"{user_points} 🌍", f"-{round((user_points - average_points) / average_points * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        else:
            st.metric("Your points", f"{user_points} 🌍", f"Average", delta_color="off", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
    
    with colb:
        # add a st.metric to show the user's number of query
        if user_num_query > average_query:
            st.metric("Eco-friendly queries", f"{user_num_query} 🌿", f"{round((user_num_query - average_query) / average_query * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        elif user_num_query < average_query:
            st.metric("Eco-friendly queries", f"{user_num_query} 🌿", f"{round((user_num_query - average_query) / average_query * 100)} %", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")
        else:
            st.metric("Eco-friendly queries", f"{user_num_query} 🌿", f"Average", delta_color="off", help="Accumulate sustainability points by giving feedback to the LLM's responses or ask a question that is already saved in the cache.")

# List of ambiguous words (this is just an example, you should expand this list)
ambiguous_words = set(["thing", "stuff", "various", "many", "often", "frequently"])

# Function to evaluate if a prompt contains ambiguous words
def has_ambiguous_words(prompt):
    words = set(re.findall(r'\b\w+\b', prompt.lower()))
    return any(word in ambiguous_words for word in words)

# Function to evaluate the prompt
def evaluate_prompt(prompt):
    criteria = {
        "uses_renewable_energy": True,  # Example: determine based on user's location or choice
        # "uses_smallest_model": st.session_state["openai_model"] == "gpt-3.5-turbo",
        "length_under_500": len(prompt) < 500,
        "no_ambiguous_words": not has_ambiguous_words(prompt), 
        "no_need_for_clarification": True  # TODO: implement logic to determine this
    }
    # Logic to update other criteria goes here
    return criteria

# Update the checklist based on the prompt
def update_checklist(prompt):
    st.session_state['checklist'] = evaluate_prompt(prompt)

# Display the checklist in the sidebar
def display_checklist():
    st.sidebar.title("Eco prompt checklist", help="This checklist is used to evaluate the sustainability of your prompt after you input it.")
    for criteria, is_met in st.session_state['checklist'].items():
        icon = "✅" if is_met else "⬜"
        st.sidebar.write(f"{icon} {criteria.replace('_', ' ').capitalize()}")
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
                    "description": f"Query string to search the posts for, inferred from the user message and ALWAYS translated to ENGLISH before passing on to the function. Sample query styles: 'expected development of stainless steel market pricing in December 2022' or 'possible energy price developments over January-March 2023'. Note: today is {dt.datetime.utcfromtimestamp(UTC_TIMESTAMP).strftime('%Y-%m-%d')}. You can use this information in your query to absolute dates instead of relative ones."
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

# st.title("PR🌍MPTERRA")

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

    # Initialize session state for metrics
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = load_all_from_mongo(username)

    # create two cols
    col41, col42 = st.sidebar.columns(2)
    add_metrics(col41, col42, username)

    # add refresh button to reload the mongo db
    st.sidebar.button("Refresh points", type="primary", on_click=refresh_metrics(), use_container_width=True)

    # st.sidebar.markdown(f"""
    #                     <p style='font-family': Garet'>Hello, {name.split()[0]}! <p> <br>
    #                     <p style='font-family': Garet'>Your points: {user_points}</p>
    #                     """, unsafe_allow_html=True)
                        

    # rewrite st info with html font family Garet
    st.markdown("""
                <p style='font-family': Garet'>Welcome to <b>PROMPTERRA</b> by <b>ECOGNIZE</b> 🌍</p> <br>
                <p style='font-family': Garet'>PROMPTERRA is a platform that trains users to use OpenAI's GPT in a more sustainable way. To get started, type a prompt in the chat box on the left and click enter. The AI will respond with a summary of your prompt. You can then provide feedback on the response to gain points!</p>
                """, unsafe_allow_html=True)

    feedback = None

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": RETRIEVAL_PROMPT}]
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "cache" not in st.session_state:
        st.session_state.cache = initialize_cache()
    # Initialize session state for the checklist
    if 'checklist' not in st.session_state:
        st.session_state['checklist'] = {
            "uses_renewable_energy": False,
            "uses_smallest_model": False,
            "length_under_500": False,
            "no_ambiguous_words": False,
            "no_need_for_clarification": False
        }

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Please provide extra information",
        "on_submit": _submit_feedback,
    }

    # Assign IDs to existing messages if they don't have one
    for n, msg in enumerate(st.session_state["messages"]):
        if msg["role"] == "system":
            continue
        contents = msg["content"]
        sources = ""
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if "SOURCES:" in contents:
                    contents, sources = contents.split("SOURCES:", 1)
                    # Clean up the sources string
                    sources = sources.strip()
                    if len(sources.split("}", 1)) == 2:
                        sources, contents_post = sources.split("}", 1)
                        sources += "}"
                        contents += f"\n{contents_post}"
            st.markdown(contents)
            if len(sources) > 0:
                try:
                    sources = json.loads(sources)
                    other_sources = []
                    for source in sources["sources"]:
                        if source.startswith("http") and source.endswith(".mp4"):
                            st.video(source)
                        else:
                            other_sources.append(source)
                    with st.expander("Sources"):
                        # Turn those sources into download links which we have the file on
                        for source_file in other_sources:
                            if source_file.startswith("http"):
                                st.link_button(source_file, source_file)
                            else:
                                st.text(source_file)
                except Exception as e:
                    st.warning(f"Error parsing sources {sources}: {e}")
                    # Display raw sources
                    with st.expander("Sources"):
                        st.markdown(sources)

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
                update_user(username, 2, 0)
                # add a notification that the user has earned a point
                st.sidebar.success(
                    f"You have earned +2 points for giving feedback!"
                )

    display_checklist()
    
    # Save cache locally to CSV (if it has a length > 0)
    if "cache" in st.session_state and len(st.session_state.cache) > 0:
        with st.sidebar:
            st.write("Cache")
            st.dataframe(st.session_state.cache)
        st.session_state.cache.to_csv("cache.csv", index=False)

    if prompt := st.chat_input("Ask me anything"):

        # Add the user message to the session state and render it
        st.session_state['messages'].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        update_checklist(prompt)
        display_checklist()

        with st.chat_message("assistant"):
            # For streaming, we need to loop through the response generator
            reply_text = ""
            function_name = ""
            function_args = ""
            for chunk in openai.ChatCompletion.create(
                # model="gpt-35-turbo-16k",
                deployment_id="gpt-35-turbo-16k",
                messages=st.session_state['messages'],
                max_tokens=1000,
                timeout=TIMEOUT,
                function_call="auto",
                functions=functions,
                stream=True,
            ):
                if len(chunk["choices"]) == 0:
                    continue
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

            # Collect full function call
            if function_name != "" and function_args != "":
                function_call = {"name": function_name, "arguments": function_args}
            else:
                function_call = None

            if function_call is None:  # Not a function call, return normal message
                # Sanitize
                if reply_text.startswith("AI: "):
                    reply_text = reply_text.split("AI: ", 1)[1]

                st.markdown(reply_text)

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
                if query is not None:
                    # Vectorize the query
                    query_embedding = embeddings.embed_query(query)

                    temp_cache = st.session_state.cache.copy()
                    # Filter out expired cache entries
                    temp_cache = temp_cache[temp_cache["expires_at"] > UTC_TIMESTAMP]
                    # Update the cache
                    st.session_state.cache = temp_cache
                    # Loop through the cache and calculate the cosine similarity between the query embedding and each of the cached embeddings
                    try:
                        temp_cache["similarity"] = temp_cache["embedding"].apply(lambda x: np.dot(np.array(eval(x)), np.array(query_embedding)) / (np.linalg.norm(np.array(eval(x))) * np.linalg.norm(np.array(query_embedding))))
                    except: # x might already be a list
                        temp_cache["similarity"] = temp_cache["embedding"].apply(lambda x: np.dot(np.array(x), np.array(query_embedding)) / (np.linalg.norm(np.array(x)) * np.linalg.norm(np.array(query_embedding))))
                    # Sort the cache by similarity, descending
                    temp_cache = temp_cache.sort_values(by=["similarity"], ascending=False)
                    # See if the top result is above a certain threshold
                    if len(temp_cache) > 0 and temp_cache.iloc[0]["similarity"] > CACHE_SIMILARITY_THRESHOLD:
                        # Directly add that answer as the chat response
                        reply_text = temp_cache.iloc[0]["answer"]
                        st.session_state['messages'].append({"role": "assistant", "content": f"(Cached Answer)\n\n{reply_text}"})
                        st.rerun()

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
                message_placeholder = st.empty()
                reply_text = ""
                for chunk in openai.ChatCompletion.create(
                    # model="gpt-4",
                    deployment_id="gpt-4",
                    messages=messages,
                    max_tokens=1500,
                    timeout=TIMEOUT,
                    # function_call="auto",
                    # functions=[],
                    stream=True,
                ):
                    if len(chunk["choices"]) == 0:
                        continue
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

                    render_text = reply_text
                    if "SOURCES:" in render_text:
                        render_text, sources = render_text.split("SOURCES:", 1)
                
                    # Continuously write the response in Streamlit
                    message_placeholder.markdown(render_text)

                # # Collect full function call
                # if function_name != "" and function_args != "":
                #     function_call = {"name": function_name, "arguments": function_args}
                # else:
                #     function_call = None
                    
                # Sanitize
                if reply_text.startswith("AI: "):
                    reply_text = reply_text.split("AI: ", 1)[1]

                render_text = reply_text
                if "SOURCES:" in render_text:
                    render_text, sources = render_text.split("SOURCES:", 1)

                message_placeholder.markdown(render_text)

                logged_prompt = collector.log_prompt(
                    config_model={"model": "gpt-4"},
                    prompt=prompt,
                    generation=reply_text,
                    session_id=st.session_state.session_id,
                    # tags=tags,
                    user_id=str(st.secrets["TRUBRICS_EMAIL"])
                    )
                
                # Add the query, its embedding, the answer (reply_text) and the expiration date to the cache. Expires in 1 day (5 minutes for testing)
                st.session_state.cache = pd.concat([
                    st.session_state.cache,
                    pd.DataFrame(
                        {
                            "query": [query],
                            "embedding": [query_embedding],
                            "answer": [reply_text],
                            "expires_at": [UTC_TIMESTAMP + 5 * 60]
                        }
                    )
                ])

                # st.session_state.prompt_ids.append(logged_prompt.id)

        # After getting the response, add it to the session state
        st.session_state['messages'].append({"role": "assistant", "content": reply_text})

        st.rerun()

    # should be the end of the sidebar
    with st.sidebar:
        authenticator.logout("Logout", "main", key="unique_key")

elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")

elif st.session_state["authentication_status"] is None:
    st.info("Please enter your username and password")

    # if feedback:
    #     st.write(feedback)