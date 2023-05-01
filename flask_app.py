
# coding: utf-8

# In[ ]:


import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap


# Set Slack API credentials
SLACK_BOT_TOKEN = "xoxb-5191045347955-5190276386373-gqUDSHZaHnCyIv178VlOCade"
SLACK_SIGNING_SECRET = "17b921377f198ddeba51bc457fff862b"
slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
response = slack_client.auth_test()
SLACK_BOT_USER_ID =response["user_id"] 

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

def vector_database(article_url):
    loader = UnstructuredURLLoader(urls=article_url)
    data = loader.load()

    text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_split.split_documents(data)

    database = FAISS.from_documents(docs, embeddings)
    return database


def query_retrive(database, query):

    docs = database.similarity_search(query, k=5)
    page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(openai_api_key="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",model_name="gpt-3.5-turbo",temperature=0.2)


    template = """
        You are a youtube video assistant that can answer questions about paragraphs feed:{docs}.
        
        Only use information from the article to answer the question.
        
        Your answers should be detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    result = chain.run(question=query, docs=page_content)
    #result = result.replace("\n", "")
    return result, docs

@app.event("app_mention")
def handle_mentions(body, say):
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    article_url = text.replace(mention, "").strip()

    say("Sure, I'll get right on that!")
    db = vector_database(article_url)
    query = "give a summary of this article?"
    response, docs = query_retrive(db, query)
    
    say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.
    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()

