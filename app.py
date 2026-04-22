"""
Interactive CLI Chat Application for RCAgentX

Provides a command-line interface for interacting with the AIOps agents.
Supports natural language queries about incidents, metrics, logs, and more.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config.settings import Settings


class AgentChat:
    """
    Interactive chat interface for AIOps agents.

    Allows users to have conversations with the AIOps system,
    ask questions about incidents, and get recommendations.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the chat interface.

        Args:
            verbose (bool): Enable verbose output
        """
        self.verbose = verbose
        self.settings = Settings.from_env()

        # Initialize LLM with conversation history
        self.llm = ChatOpenAI(
            model=self.settings.llm.model,
            api_key=self.settings.llm.api_key,
            base_url=self.settings.llm.api_base,
            temperature=0.7,
        )

        # Conversation history
        self.history = []

        # System prompt for the AIOps assistant
        self.system_prompt = """You are an AIOps (Artificial Intelligence for IT Operations) assistant.
Your role is to help users with:
1. Analyzing incidents and anomalies
2. Root cause analysis
3. Remediation recommendations
4. Monitoring and observability questions
5. Best practices for SRE and operations

You are knowledgeable about:
- Prometheus metrics and alerting
- Log analysis (Loki, ELK)
- Kubernetes operations
- Distributed tracing
- Incident response workflows

Be concise, practical, and provide actionable advice.
When you don't know something, admit it and suggest where to find more information."""

    def _get_system_message(self) -> SystemMessage:
        """Create system message with context"""
        return SystemMessage(content=self.system_prompt)

    def chat(self, user_input: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_input (str): User's message

        Returns:
            str: AI assistant's response
        """
        # Build messages with history
        messages = [self._get_system_message()]

        # Add recent history (last 10 messages)
        for msg in self.history[-10:]:
            messages.append(msg)

        # Add current message
        messages.append(HumanMessage(content=user_input))

        # Get response
        response = self.llm.invoke(messages)

        # Update history
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))

        return response.content

    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        print("\n[System] Conversation history cleared.\n")

    def show_history(self, limit: int = 5):
        """Show recent conversation history"""
        if not self.history:
            print("\n[System] No conversation history.\n")
            return

        print("\n" + "=" * 60)
        print("Recent Conversation")
        print("=" * 60)

        start = max(0, len(self.history) - limit * 2)
        for i in range(start, len(self.history), 2):
            if i + 1 < len(self.history):
                user_msg = self.history[i].content
                ai_msg = self.history[i + 1].content
                print(f"\nYou: {user_msg[:100]}..." if len(user_msg) > 100 else f"\nYou: {user_msg}")
                print(f"AI: {ai_msg[:100]}..." if len(ai_msg) > 100 else f"AI: {ai_msg}")

        print("=" * 60 + "\n")


def print_welcome():
    """Print welcome message and help information"""
    print("""
╔════════════════════════════════════════════════════════╗
║          RCAgentX - AIOps Interactive Assistant        ║
╠════════════════════════════════════════════════════════╣
║  Type your questions and press Enter to chat.          ║
║                                                         ║
║  Commands:                                               ║
║    /help     - Show this help message                   ║
║    /clear    - Clear conversation history               ║
║    /history  - Show recent conversation                 ║
║    /status   - Show system status                       ║
║    /quit     - Exit the application                     ║
║    /exit     - Exit the application                     ║
║                                                         ║
║  Example questions:                                      ║
║    "What is AIOps?"                                     ║
║    "How do I troubleshoot high CPU usage?"              ║
║    "What are the best practices for alerting?"          ║
║    "Explain root cause analysis"                        ║
╚════════════════════════════════════════════════════════╝
""")


def print_status(chat: AgentChat):
    """Print system status"""
    print("\n" + "=" * 40)
    print("System Status")
    print("=" * 40)
    print(f"LLM Model: {chat.settings.llm.model}")
    print(f"LLM Base URL: {chat.settings.llm.api_base or 'default'}")
    print(f"Conversation turns: {len(chat.history) // 2}")
    print(f"Prometheus URL: {chat.settings.prometheus.url}")
    print(f"Loki URL: {chat.settings.loki.url}")
    print("=" * 40 + "\n")


def main():
    """Main entry point for the interactive chat"""
    print_welcome()

    # Initialize chat
    try:
        chat = AgentChat(verbose=os.getenv("VERBOSE", "false").lower() == "true")
        print("[System] Connected to AIOps assistant.\n")
    except Exception as e:
        print(f"\n[Error] Failed to initialize: {e}")
        print("\nMake sure your API key is valid and network is available.")
        sys.exit(1)

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\033[92mYou:\033[0m ").strip()

            # Empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd in ["/quit", "/exit"]:
                    print("\n[System] Goodbye!\n")
                    break

                elif cmd == "/help":
                    print_welcome()

                elif cmd == "/clear":
                    chat.clear_history()

                elif cmd == "/history":
                    chat.show_history()

                elif cmd == "/status":
                    print_status(chat)

                else:
                    print(f"\n[System] Unknown command: {user_input}")
                    print("Type /help for available commands.\n")

                continue

            # Normal chat - get AI response
            print("\033[94mAI:\033[0m ", end="", flush=True)

            # Stream response (character by character for effect)
            response = chat.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n[System] Interrupted. Type /quit to exit.\n")
        except EOFError:
            print("\n\n[System] Goodbye!\n")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()
