from openai import OpenAI
from dataclasses import dataclass, field
from typing import List
import tiktoken

@dataclass
class Message:
    role: str
    content: str
    tokens: int = 0

@dataclass
class ConversationStats:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    message_count: int = 0

class CostAwareChat:
    """Chat application with cost tracking and optimization"""
    
    def __init__(self, model: str = "gpt-4o-mini", budget_limit: float = 1.0):
        self.client = OpenAI()
        self.model = model
        self.budget_limit = budget_limit
        self.messages: List[Message] = []
        self.stats = ConversationStats()
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        
        # Pricing per million tokens
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
    
    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.pricing[self.model]
        return (input_tokens / 1_000_000) * pricing["input"] + \
               (output_tokens / 1_000_000) * pricing["output"]
    
    def _build_messages(self) -> list:
        """Build message list, potentially summarizing old messages"""
        total_tokens = sum(m.tokens for m in self.messages)
        
        # If approaching context limit, summarize old messages
        if total_tokens > 10000:
            return self._summarize_and_build()
        
        return [{"role": m.role, "content": m.content} for m in self.messages]
    
    def _summarize_and_build(self) -> list:
        """Summarize conversation history to save tokens"""
        # Keep last 5 messages, summarize the rest
        recent = self.messages[-5:]
        old = self.messages[:-5]
        
        if old:
            old_text = "\n".join([f"{m.role}: {m.content}" for m in old])
            summary_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"Summarize this conversation in 2-3 sentences:\n{old_text}"
                }],
                max_tokens=200
            )
            summary = summary_response.choices[0].message.content
            
            return [
                {"role": "system", "content": f"Previous conversation summary: {summary}"},
                *[{"role": m.role, "content": m.content} for m in recent]
            ]
        
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def chat(self, user_message: str) -> str:
        """Send message and get response with cost tracking"""
        # Check budget
        if self.stats.total_cost >= self.budget_limit:
            return f"Budget limit of ${self.budget_limit} reached. Total spent: ${self.stats.total_cost:.4f}"
        
        # Add user message
        user_tokens = self._count_tokens(user_message)
        self.messages.append(Message("user", user_message, user_tokens))
        
        # Build and send
        messages = self._build_messages()
        input_tokens = sum(self._count_tokens(m["content"]) for m in messages)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        assistant_content = response.choices[0].message.content
        output_tokens = self._count_tokens(assistant_content)
        
        # Track stats
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens
        self.stats.total_cost += self._calculate_cost(input_tokens, output_tokens)
        self.stats.message_count += 1
        
        # Add assistant message
        self.messages.append(Message("assistant", assistant_content, output_tokens))
        
        return assistant_content
    
    def get_stats(self) -> dict:
        """Get conversation statistics"""
        return {
            "messages": self.stats.message_count,
            "input_tokens": self.stats.total_input_tokens,
            "output_tokens": self.stats.total_output_tokens,
            "total_cost": f"${self.stats.total_cost:.6f}",
            "budget_remaining": f"${self.budget_limit - self.stats.total_cost:.6f}",
            "model": self.model
        }


# Usage
chat = CostAwareChat(model="gpt-4o-mini", budget_limit=0.50)

print(chat.chat("What is machine learning?"))
print(chat.chat("Give me an example"))
print(chat.chat("How is it different from AI?"))

print("\n--- Stats ---")
print(chat.get_stats())