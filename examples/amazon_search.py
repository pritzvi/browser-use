"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
logger = logging.getLogger(__name__)


from browser_use import Agent

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=4,
)

agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)



async def main():
	await agent.run(max_steps=10)
	input('Press Enter to continue...')


asyncio.run(main())
