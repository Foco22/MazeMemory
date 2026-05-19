from src.maze.instances import get_maze, available_maze_ids
from src.maze.pathfinding import optimal_steps

for maze_id in available_maze_ids():
    maze = get_maze(maze_id)
    print(f"=== Maze {maze_id} (seed={maze.seed}) ===")
    print(maze.render())
    print()
    for i, start in enumerate(maze.start_positions):
        steps = optimal_steps(maze, start, maze.exit_pos)
        print(f"  Agent {i+1} {start} → E{maze.exit_pos}  optimal: {steps} steps")
    print()



import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "google/gemma-4-31b-it",
  "messages": [{"role":"user","content":"hi"}],
  "max_tokens": 16384,
  "temperature": 1.00,
  "top_p": 0.95,
  "stream": stream,
  "chat_template_kwargs": {"enable_thinking":True},
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
