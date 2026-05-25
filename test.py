import asyncio, litellm

tools = [{
    "type": "function",
    "function": {
        "name": "move",
        "description": "Move in a direction",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["north", "south", "east", "west"]}
            },
            "required": ["direction"]
        }
    }
}]

async def test():
    r = await litellm.acompletion(
        model="openai/ZimaBlueAI/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF:latest",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        messages=[{"role": "user", "content": "Move north please"}],
        tools=tools,
    )
    msg = r.choices[0].message
    print("tool_calls:", msg.tool_calls)
    print("content:", msg.content)

asyncio.run(test())
