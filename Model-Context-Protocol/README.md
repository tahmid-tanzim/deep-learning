# Model Context Protocol


### Launch MCP Inspector
- Install dependencies
```
uv init
uv add -r requirements.txt
```

- Make sure node version 18 is installed
```
nvm install 18
nvm use 18
```

- Run MCP Inspector
`npx @modelcontextprotocol/inspector uv run Model-Context-Protocol/mcp-research-server.py`

- Run MCP Client with stdio server
`uv run Model-Context-Protocol/mcp-chatbot.py`

- Run MCP Client with multiple resource server
`uv run Model-Context-Protocol/mcp-chatbot-connected-to-resource-server.py`