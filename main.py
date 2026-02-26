import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# --- 1. Security & API Keys ---
# This line loads your local .env file when you test on your computer.
# When running live on Render, Render ignores this and uses the dashboard variables you set.
load_dotenv()

# A quick safety check so the server crashes loudly if the key is missing, 
# rather than failing silently later.
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY is missing! Please add it to your .env file or Render dashboard.")

# --- 2. Initialize Server & Security Rules ---
app = FastAPI()

# CORS restricts who can talk to your AI. We are allowing your live domain 
# and localhost (so you can test your React frontend on your computer).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://drhoneypalani.com", 
        "https://www.drhoneypalani.com",
        "http://localhost:3000",
        "http://localhost:5173" # Adding common React local ports just in case
    ], 
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- 3. Define Farm Tools ---
@tool
def check_inventory(product_name: str) -> str:
    """Checks if a specific honey product is in stock. Use for queries about availability."""
    if "comb" in product_name.lower():
        return "We have 10 jars of raw Comb Honey in stock."
    elif "moringa" in product_name.lower():
        return "We have 25 jars of Moringa Honey in stock."
    else:
        return f"Let me check the hives! I don't see {product_name} in our current inventory."

@tool
def get_academy_schedule(course_type: str) -> str:
    """Fetches upcoming dates and prices for Hands-On Training Academy courses."""
    if "basic" in course_type.lower():
        return "The next 1-Day Basic Beekeeping course is on March 15th. It costs ₹1,500."
    elif "commercial" in course_type.lower():
        return "The next 3-Month Commercial Apiculture training starts on April 1st. It costs ₹15,000."
    else:
        return "Please specify if you are interested in Basic Beekeeping or Commercial Apiculture."

# --- 4. Initialize the AI Brain ---
tools = [check_inventory, get_academy_schedule]
# The LLM will automatically find the GOOGLE_API_KEY in the environment variables
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent_executor = create_react_agent(llm, tools)

# --- 5. Define the Chat Endpoint ---
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    # Pass the user's message into the LangGraph loop
    inputs = {"messages": [("user", request.message)]}
    result = agent_executor.invoke(inputs)
    
    # Grab the final text response from the AI
    final_reply = result["messages"][-1].content
    
    return {"reply": final_reply}