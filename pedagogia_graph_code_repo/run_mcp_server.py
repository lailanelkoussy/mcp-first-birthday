import os
import sys
from dotenv import load_dotenv
from starlette.responses import JSONResponse
from langfuse import get_client

from RepoKnowledgeGraphLib.KnowledgeGraphMCPServer import KnowledgeGraphMCPServer

load_dotenv()

langfuse = get_client()

def main():
    print("🚀 Starting Knowledge Graph MCP Server...", file=sys.stderr, flush=True)

    model_service_kwargs = {
        "embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
    }
    code_index_kwargs= {
        "index_type": "keyword-only",
    }
    
    """
    mcp_server = KnowledgeGraphMCPServer.from_file(
        filepath=os.getenv('FILE_NAME'),
        index_nodes=True,
        use_embed=True,
        model_service_kwargs=model_service_kwargs,
    code_index_kwargs=code_index_kwargs)
    
    """
    
    if os.getenv("REPO_URL"):
        # Create your KnowledgeGraphMCPServer instance
        mcp_server = KnowledgeGraphMCPServer.from_repo(
            repo_url=os.getenv("REPO_URL"),
            index_nodes=True,
            describe_nodes=False,
            extract_entities=True,
            model_service_kwargs=model_service_kwargs,
            github_token=os.getenv('GITHUB_TOKEN', None), 
            code_index_kwargs=code_index_kwargs
        )
    elif os.getenv('REPO_PATH'):
        # Create your KnowledgeGraphMCPServer instance
        mcp_server = KnowledgeGraphMCPServer.from_path(
            path=os.getenv("REPO_PATH"),
            index_nodes=True,
            describe_nodes=False,
            extract_entities=True,
            model_service_kwargs=model_service_kwargs, 
            code_index_kwargs=code_index_kwargs
        )
        
    else:    
        print("❌ Error: Please set either REPO_URL or REPO_PATH environment variable.", file=sys.stderr, flush=True)
        sys.exit(1)



    # Add health check endpoint
    @mcp_server.app.custom_route("/health", methods=["GET"])
    async def health_check(request):
        return JSONResponse({"status": "healthy", "service": "mcp-server"})

    print("✅ MCP Server initialized successfully", file=sys.stderr, flush=True)

    # Run the MCP server directly over HTTP
    port = int(os.environ.get("PORT", 4000))
    mcp_server.app.run(
        transport="http",
        host="0.0.0.0",
        port=port,
        path="/mcp"
    )


if __name__ == "__main__":
    main()