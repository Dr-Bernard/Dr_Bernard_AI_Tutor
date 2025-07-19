import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ChatAgent:
    def __init__(self, api_key, model_name, verbose=False):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.conversation_history = []
        self.verbose = verbose

    def web_search(self, query):
        subscription_key = os.environ.get("TAVILY_API_KEY")
        search_url = "https://api.tavily.com/v1/search"
        headers = {"Authorization": f"Bearer {subscription_key}"}
        params = {"query": query, "textDecorations": True, "textFormat": "HTML"}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        return search_results["webPages"]["value"][0]["snippet"]

    def get_response(self, user_input):
        if self.verbose:
            print(f"User input received: {user_input}")
            print("Appending user input to conversation history...")
        
        self.conversation_history.append({"role": "user", "content": user_input})

        # Check if the user input indicates a need for real-time information
        if "search" in user_input.lower():
            if self.verbose:
                print("Detected need for real-time information. Performing web search...")
            search_query = user_input.lower().replace("search", "").strip()
            web_search_result = self.web_search(search_query)
            self.conversation_history.append({"role": "assistant", "content": web_search_result})
            if self.verbose:
                print(f"Web search result: {web_search_result}")
            return web_search_result

        if self.verbose:
            print("Sending conversation history to Groq API for response...")
        
        chat_completion = self.client.chat.completions.create(
            messages=self.conversation_history,
            model=self.model_name,
        )

        response = chat_completion.choices[0].message.content
        
        if self.verbose:
            print(f"Response received: {response}")
            print("Appending assistant response to conversation history...")

        self.conversation_history.append({"role": "assistant", "content": response})
        
        if self.verbose:
            print("Returning the response to the user.")

        return response

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in .env file.")
        return

    model_name = "llama3-8b-8192"
    agent = ChatAgent(api_key, model_name, verbose=True)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        response = agent.get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
