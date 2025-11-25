#!/bin/bash

# Stop Script for Smolagent Chat Setup

set -e

echo "🛑 Stopping Knowledge Graph Chat Agent services..."
echo ""

docker-compose -f docker-compose-chat.yml down

echo ""
echo "✅ All services stopped"
echo ""
echo "💡 Tips:"
echo "   • To view logs later: docker-compose -f docker-compose-chat.yml logs"
echo "   • To start again: ./start-chat.sh"
echo "   • To remove volumes: docker-compose -f docker-compose-chat.yml down -v"
echo ""
