"""
Provider selection example.

Shows how to point KSS RAG at different LLM providers — hosted, local, and
custom — using the create_llm factory. All adapters share one interface, so the
rest of your code stays identical regardless of provider.

Run:  python examples/providers.py
"""
from kssrag import create_llm, list_providers


def main():
    print("Available providers:")
    for name in sorted(list_providers()):
        preset = list_providers()[name]
        key = preset.api_key_env or "(no key)"
        print(f"  {name:16} kind={preset.kind:10} key={key}")

    # Build an adapter for a few providers (no network call is made here).
    examples = [
        dict(provider="openrouter", model="deepseek/deepseek-chat-v3.1:free"),
        dict(provider="groq", model="llama-3.3-70b-versatile"),
        dict(provider="ollama", model="llama3"),           # local, no key
        dict(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-ant-..."),
        dict(provider="custom", base_url="http://localhost:8000/v1/chat/completions",
             model="my-model"),
    ]

    print("\nConstructed adapters:")
    for kwargs in examples:
        try:
            llm = create_llm(**kwargs)
            print(f"  {kwargs['provider']:12} -> {type(llm).__name__}")
        except ValueError as e:
            # e.g. a hosted provider with no key configured in this environment
            print(f"  {kwargs['provider']:12} -> needs config: {e}")

    # Usage is identical no matter which adapter you built:
    #   answer = llm.predict([{"role": "user", "content": "Hello"}])
    #   for token in llm.predict_stream(messages): ...


if __name__ == "__main__":
    main()
