# Sumo Handicapper - Claude Instructions

## Database Context

This project uses the **dewsweeper3** MySQL database accessed via Google Cloud SQL Proxy.

**When to use the MCP dewsweeper3-mysql server:**
- When the user asks about data, statistics, or historical information about sumo wrestling
- When queries mention sumo wrestlers, bouts, tournaments, basho, rankings, or match results
- When implementing features that need to fetch or analyze sumo data
- When the user explicitly mentions "dewsweeper" or "dewsweeper3"
- When working with sumo-related database queries in this project

**Connection details:**
- Database: dewsweeper3 (MySQL)
- Access: Via MCP server "dewsweeper3-mysql"
- Contains: Sumo wrestling historical data including wrestlers, bouts, tournaments, and rankings

Use the MCP server's schema inspection tools to understand table structures when working with database queries.

## MCP Server Setup

To set up the MCP server on a new machine:

1. **Create `.env` file** (git-ignored) with database credentials:
   ```env
   DB_HOST=127.0.0.1
   DB_PORT=3307
   DB_USER=dewsweeper
   DB_NAME=dewsweeper3
   DB_PASSWORD=your_password_here
   ```

2. **Configure Claude Desktop MCP** by adding to `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "dewsweeper3-mysql": {
         "command": "uvx",
         "args": [
           "mcp-server-mysql",
           "--host", "127.0.0.1",
           "--port", "3307",
           "--user", "dewsweeper",
           "--password", "YOUR_PASSWORD_HERE",
           "--database", "dewsweeper3"
         ]
       }
     }
   }
   ```

3. **Restart Claude Desktop** to load the MCP server

4. **Ensure Cloud SQL Proxy is running** to connect to the database

## Running the Application

To run the Streamlit prediction app (works on both Windows and Mac):

```bash
uv run streamlit run src/prediction/streamlit_app.py
```

This command works cross-platform because `uv` handles the virtual environment activation automatically.
