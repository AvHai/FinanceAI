from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os

from dotenv import load_dotenv
load_dotenv()

#Agent 1
websearch_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information", ## prompt 
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),## model
    tools=[DuckDuckGo()],
    instructions=["Include Sources"],
    show_tool_calls=True,
    markdown=True
)

##Agent 2
finance_agent=Agent(
    name="Finance AI Agent",
    role="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use Tables to display the data of the stock using markdown"],
    show_tool_calls=True,
    markdown=True
)


multi_ai_agent=Agent(
    team=[websearch_agent,finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
