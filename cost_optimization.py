"""
Cost Optimization Patterns
Reducing LLM costs in production
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import traceable

load_dotenv()


class TokenBudget:
    """Track and limit token per usage."""

    def __init__(self, max_tokens_per_request: int = 400):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_token(slef, text: str) -> int:
        """Rough token estimate"""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """check if request is within budget."""
        tokens = self.estimate_token(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage"""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }


class BudgetedLLM:
    """LLM with token budgeting"""

    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:
        # check budget
        within_budget, token = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {token} > {self.budget.max_per_request}"
            )

        # execute
        response = self.llm.invoke(query)
        result = response.content

        # record output
        output_tokens = self.budget.estimate_token(result)
        self.budget.record_usage(token, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    """Demonstrate token budgeting"""

    llm = BudgetedLLM(max_tokens=100)

    queries = ["what is ATS", "Explain " + "very " * 100 + "complex topic"]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"✅ {query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌ {query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")


if __name__ == "__main__":
    demo_token_budgeting()
