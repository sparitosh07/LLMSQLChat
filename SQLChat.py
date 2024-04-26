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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
import getpass
import os
# import examples






def init_database(database: str) -> SQLDatabase:
  db_uri = r"sqlite:///C:\Users\Paritosh\Desktop\AI\LLM\prop_insurance_database.db"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  examples =[
     {
         "input": "How many policies are inforce for Q1 2024",
         "query": "SELECT count(*) FROM policies WHERE start_date >= '2024-01-01' "},
     {
         "input": "Get the policy with the highest loss",
         "query": "SELECT p.policy_id, SUM(claim_amount) as ClaimAmount from policies p join claims c on p.policy_id = c.policy_id group by p.policy_id order by SUM(claim_amount) desc limit 1"   
     }
     ]
  
  example_prompt = ChatPromptTemplate.from_messages(
     [
         ("human", "{input}\nSQLQuery:"),
         ("ai", "{query}"),
     ])
    
  few_shot_prompt = FewShotChatMessagePromptTemplate(
     example_prompt=example_prompt,
     examples=examples
     # input_variables=["input","top_k"],
    #  input_variables=["input"],
 )

  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    Your turn:
    
    Question: {input}
    SQL Query:
    """
  sql_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', template),
        few_shot_prompt,
        ('human', '{input}'),
    ]
    )
#   sql_prompt = ChatPromptTemplate.from_template(template)
  llm = ChatCohere(cohere_api_key="4y2CwVZz8ocMUXaA37mDGDKwbmIIhR6SztGEpnCS")
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
    Based on the table schema below, question, sql query, and sql response, write a natural language response. Provide the output in a tabular format
    wherever possible.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>`
    User question: {input}
    SQL Response: {response}"""
  
  answer_prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
#   llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  llm = ChatCohere(cohere_api_key="4y2CwVZz8ocMUXaA37mDGDKwbmIIhR6SztGEpnCS")
  
  answer = answer_prompt | llm | StrOutputParser()

  chain = (RunnablePassthrough.assign(query=sql_chain).assign( schema=lambda _: db.get_table_info(),response=itemgetter("query") | execute_query) | answer)
  
  return chain.invoke({
    "input": user_query,
    "chat_history": chat_history,
  })
    

def get_response_exmples(user_query: str, db: SQLDatabase, chat_history: list):
    
    sql_chain = get_sql_chain(db)
    execute_query = QuerySQLDataBaseTool(db=db)
    
    final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful AI Assistant'),
        few_shot_prompt,
        ('human', '{input}'),
    ]
)
    llm = ChatCohere(cohere_api_key="4y2CwVZz8ocMUXaA37mDGDKwbmIIhR6SztGEpnCS")
    answer = final_prompt | llm | StrOutputParser()
    chain = (RunnablePassthrough.assign(query=sql_chain).assign( schema=lambda _: db.get_table_info(),response=itemgetter("query") | execute_query) | answer)
  
    



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