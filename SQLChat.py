import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.messages import AIMessage, HumanMessage
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import getpass
import os






def init_database(database: str) -> SQLDatabase:
  db_uri = r"sqlite:///C:\Users\Paritosh\Desktop\AI\LLM\prop_insurance_database.db"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  sql_prompt = ChatPromptTemplate.from_template(template)
  llm = ChatCohere(cohere_api_key="")
    #llm = ChatOpenAI(model="gpt-4-0125-preview")
    #llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
#   return (
#     create_sql_query_chain(llm, db,sql_prompt,5)
#   )
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | llm
    | StrOutputParser()
  )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
 
  sql_chain = get_sql_chain(db)
  execute_query = QuerySQLDataBaseTool(db=db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>`
    User question: {question}
    SQL Response: {response}"""
  
  answer_prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
#   llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  llm = ChatCohere(cohere_api_key="4y2CwVZz8ocMUXaA37mDGDKwbmIIhR6SztGEpnCS")
  
  answer = answer_prompt | llm | StrOutputParser()

  chain = (RunnablePassthrough.assign(query=sql_chain).assign( schema=lambda _: db.get_table_info(),response=itemgetter("query") | execute_query) | answer)
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# load_dotenv()
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("Chat with iHub")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    # st.text_input("Host", value="localhost", key="Host")
    # st.text_input("Port", value="3306", key="Port")
    # st.text_input("User", value="root", key="User")
    # st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="iHub", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                # st.session_state["User"],
                # st.session_state["Password"],
                # st.session_state["Host"],
                # st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))