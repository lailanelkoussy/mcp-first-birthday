import sys
import argparse
from dotenv import load_dotenv

from RepoKnowledgeGraphLib.KnowledgeGraphMCPServer import KnowledgeGraphMCPServer

load_dotenv()


def main():
    print("🚀 Starting Knowledge Graph MCP Server...", file=sys.stderr, flush=True)

    parser = argparse.ArgumentParser(description="Launch MCP Server for a code repository")
    parser.add_argument("--repo-path", type=str, required=True, help="Path to the code repository")
    args = parser.parse_args()

    model_service_kwargs = {
        "embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
    }

    # Create your KnowledgeGraphMCPServer instance
    mcp_server = KnowledgeGraphMCPServer.from_path(
        path=args.repo_path,
        index_nodes=True,
        describe_nodes=False,
        extract_entities=True,
        model_service_kwargs=model_service_kwargs,

    )

    print("✅ MCP Server initialized successfully", file=sys.stderr, flush=True)
    mcp_server.app.run(transport="stdio")


if __name__ == "__main__":
    main()