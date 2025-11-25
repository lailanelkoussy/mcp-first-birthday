#!/bin/bash

# Quick Start Script for Smolagent Chat Setup
# This script helps you get started with the Knowledge Graph Chat Agent

set -e

echo "🚀 Knowledge Graph Chat Agent - Quick Start"
echo "==========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from template..."
    
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file from .env.example"
        echo ""
        echo "⚠️  IMPORTANT: Edit the .env file and set your OPENAI_API_KEY"
        echo "   nano .env   (or use your preferred editor)"
        echo ""
        read -p "Press Enter after you've set your API key in .env..."
    else
        echo "❌ Error: .env.example not found!"
        exit 1
    fi
fi

# Check if OPENAI_API_KEY is set in .env
if grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env 2>/dev/null; then
    echo "⚠️  Warning: OPENAI_API_KEY appears to be the default value"
    echo "   Please edit .env and set a valid API key"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🐳 Starting Docker services..."
echo ""

# Build and start services
docker-compose -f docker-compose-chat.yml up --build -d

echo ""
echo "⏳ Waiting for services to become healthy..."
echo "   This may take 2-5 minutes for the first run..."
echo ""

# Wait for services to be healthy
MAX_WAIT=300  # 5 minutes
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker-compose -f docker-compose-chat.yml ps | grep -q "unhealthy"; then
        echo "   Still starting... (${ELAPSED}s elapsed)"
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    elif docker-compose -f docker-compose-chat.yml ps | grep -q "(healthy)"; then
        echo "✅ All services are healthy!"
        break
    else
        echo "   Waiting for health checks... (${ELAPSED}s elapsed)"
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "⚠️  Services took longer than expected to start"
    echo "   Check the logs with: docker-compose -f docker-compose-chat.yml logs"
fi

echo ""
echo "✨ Services are ready!"
echo ""
echo "📍 Access the interfaces:"
echo "   🤖 AI Chat Interface:    http://localhost:7861"
echo "   🔧 Direct MCP Interface: http://localhost:7860"
echo "   📡 MCP Server API:       http://localhost:4000/mcp"
echo ""
echo "📊 View logs:"
echo "   docker-compose -f docker-compose-chat.yml logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose -f docker-compose-chat.yml down"
echo ""
echo "Happy exploring! 🎉"
