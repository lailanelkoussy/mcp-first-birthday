#!/usr/bin/env python3
"""
Comprehensive MCP Client Test - Simulates what an LLM would see.
Tests all aspects of the MCP protocol including discovery, execution, error handling, and chaining.
"""

import asyncio
import sys
import os
import json
from fastmcp import Client


class Colors:
    """ANSI color codes for better output readability."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_section(title, color=Colors.BLUE):
    """Print a formatted section header."""
    print(f"\n{color}{'=' * 70}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{color}{'=' * 70}{Colors.ENDC}\n")


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{Colors.CYAN}{'-' * 70}{Colors.ENDC}")
    print(f"{Colors.CYAN}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-' * 70}{Colors.ENDC}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_info(message):
    """Print an info message."""
    print(f"{Colors.YELLOW}ℹ️  {message}{Colors.ENDC}")


def pretty_print_node_info(result):
    """Nicely format get_node_info CallToolResult for readability."""
    # Try to use structured_content if available
    if hasattr(result, "structured_content") and result.structured_content and "result" in result.structured_content:
        lines = result.structured_content["result"].splitlines()
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                print(f"{Colors.BOLD}{key.strip():<15}{Colors.ENDC}: {value.strip()}")
            else:
                print(line)
    elif hasattr(result, "data"):
        # fallback to raw text
        print(result.data)
    else:
        print(str(result))


async def test_client_connection(client):
    try:
        await client.ping()
        print_success('Client connection successful')
        return True
    except Exception as e:
        print_error(f"Client connection failed: {e}")
        return False

async def test_tool_discovery(client):
    """Test 2: Tool discovery - what the LLM sees first."""
    print_section("TEST 2: Tool Discovery (LLM's First View)", Colors.HEADER)

    print_info("Calling /tools endpoint - This is how the LLM discovers what it can do")
    try:
        tools = await client.list_tools()
        print_success(f"Discovered {len(tools)} available tools\n")

        print("📋 Tool Inventory (What the LLM sees):")
        print("=" * 70)
        return True, tools
    except Exception as e:
        print_error(f"Tool discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


async def test_basic_tool_execution(client):
    """Test 3: Basic tool execution without parameters."""
    print_section("TEST 3: Basic Tool Execution (No Parameters)", Colors.HEADER)

    print_info("Testing get_graph_stats() - simplest tool call with no parameters")
    print("LLM action: call_tool('get_graph_stats', {})")

    try:
        result = await client.call_tool("get_graph_stats", {})
        response_text = result.data if hasattr(result, "data") else str(result)
        print_subsection("Server Response:")
        print(response_text)

        # Validate response structure
        checks = [
            ("Total Nodes:" in response_text, "Response contains node count"),
            ("Total Edges:" in response_text, "Response contains edge count"),
            ("Node Types:" in response_text, "Response contains node type breakdown"),
            ("Edge Relations:" in response_text, "Response contains edge relation breakdown")
        ]

        print_subsection("Response Validation:")
        all_passed = True
        for check, description in checks:
            if check:
                print_success(description)
            else:
                print_error(description)
                all_passed = False

        return all_passed, response_text

    except Exception as e:
        print_error(f"Basic tool execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_parameterized_tool_execution(client):
    """Test 4: Tool execution with parameters."""
    print_section("TEST 4: Parameterized Tool Execution", Colors.HEADER)

    test_cases = [
        {
            'tool': 'search_nodes',
            'args': {'query': 'controller', 'limit': 5},
            'description': 'Search for nodes containing "controller"'
        },
        {
            'tool': 'list_nodes_by_type',
            'args': {'node_type': 'file', 'limit': 10},
            'description': 'List file-type nodes'
        },
        {
            'tool': 'search_nodes',
            'args': {'query': 'class'},
            'description': 'Search with default limit parameter'
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        print(f"LLM action: call_tool('{test_case['tool']}', {json.dumps(test_case['args'])})")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            # Print first 500 chars to keep output manageable
            if len(response_text) > 500:
                print(response_text[:500] + "\n... (truncated)")
            else:
                print(response_text)

            print_success(f"Tool '{test_case['tool']}' executed successfully")
            results.append((True, test_case['tool'], response_text))

        except Exception as e:
            print_error(f"Tool '{test_case['tool']}' failed: {e}")
            results.append((False, test_case['tool'], str(e)))

    success_count = sum(1 for r in results if r[0])
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(test_cases)} parameterized tests passed{Colors.ENDC}")

    return results


async def test_tool_chaining(client):
    """Test 5: Tool chaining - simulating multi-step LLM reasoning."""
    print_section("TEST 5: Tool Chaining (Multi-Step LLM Workflow)", Colors.HEADER)

    print_info("Simulating how an LLM would chain multiple tool calls to answer:")
    print(f"{Colors.BOLD}User Question: 'Tell me about the controllers in the codebase'{Colors.ENDC}\n")

    # Step 1: Search for controllers
    print_subsection("LLM Step 1: Search for relevant nodes")
    print("LLM reasoning: I should search for 'controller' to find relevant code")
    print("LLM action: call_tool('search_nodes', {'query': 'controller', 'limit': 3})")

    try:
        search_result = await client.call_tool("search_nodes", {
            "query": "controller",
            "limit": 3
        })
        search_text = search_result.data if hasattr(search_result, "data") else str(search_result)
        print("\nServer Response:")
        print(search_text)
        print_success("Search completed")

        # Step 2: Extract node ID from search results
        print_subsection("LLM Step 2: Parse results and get detailed info")
        node_id = None
        node_name = None
        # Try to extract from header line like: ====================train.py_5===================
        for line in search_text.split('\n'):
            line = line.strip()
            if line.startswith('=') and line.endswith('=') and len(line) > 10:
                parts = line.strip('=').strip()
                if parts:
                    node_id = parts
                    node_name = parts
                    break
        # Fallback: try old bullet-list style
        if not node_id and "- " in search_text:
            for line in search_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        node_id = line.split('-')[1].split(':')[0].strip()
                        node_name = line.split(':')[1].split('(')[0].strip()
                        break
                    except:
                        pass

        if node_id:
            print(f"LLM reasoning: Found node '{node_name}' with ID '{node_id}', let me get more details")
            print(f"LLM action: call_tool('get_node_info', {{'node_id': '{node_id}'}})")

            info_result = await client.call_tool("get_node_info", {
                "node_id": node_id
            })
            print_subsection("Formatted Node Info:")
            pretty_print_node_info(info_result)
            info_text = info_result.data if hasattr(info_result, "data") else str(info_result)
            print_success("Got detailed node information")

            # Step 3: Get relationships
            print_subsection("LLM Step 3: Explore relationships")
            print(f"LLM reasoning: Now let me see what this node is connected to")
            print(f"LLM action: call_tool('get_node_edges', {{'node_id': '{node_id}'}})")

            edges_result = await client.call_tool("get_node_edges", {
                "node_id": node_id
            })
            edges_text = edges_result.data if hasattr(edges_result, "data") else str(edges_result)
            print("\nServer Response:")
            print(edges_text)
            print_success("Got node relationships")

            # Step 4: LLM synthesizes response
            print_subsection("LLM Step 4: Synthesize final response")
            print(f"{Colors.GREEN}LLM would now respond to user:{Colors.ENDC}")
            print(f"{Colors.BOLD}\"I found several controllers in the codebase. {node_name} is one of them.")
            print("Based on the code structure, it contains [info from node details]")
            print("and has dependencies on [info from edges]...\"{Colors.ENDC}")

            print_success("Tool chaining demonstration complete")
            return True
        else:
            print_error("Could not extract node ID from search results")
            return False

    except Exception as e:
        print_error(f"Tool chaining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling(client):
    """Test 6: Error handling and edge cases."""
    print_section("TEST 6: Error Handling & Edge Cases", Colors.HEADER)

    error_test_cases = [
        {
            'tool': 'get_node_info',
            'args': {'node_id': 'nonexistent_node_12345'},
            'description': 'Non-existent node ID',
            'expected': 'not found'
        },
        {
            'tool': 'get_node_edges',
            'args': {'node_id': 'invalid_node'},
            'description': 'Invalid node for edges query',
            'expected': 'not found'
        },
        {
            'tool': 'list_nodes_by_type',
            'args': {'node_type': 'nonexistent_type', 'limit': 5},
            'description': 'Non-existent node type',
            'expected': 'No nodes found'
        },
    ]

    results = []
    for i, test_case in enumerate(error_test_cases, 1):
        print_subsection(f"Error Test {i}: {test_case['description']}")
        print(f"LLM action: call_tool('{test_case['tool']}', {json.dumps(test_case['args'])})")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text)

            # Check if error is handled gracefully
            if test_case['expected'].lower() in response_text.lower():
                print_success("Error handled gracefully with informative message")
                results.append(True)
            else:
                print_error(f"Expected '{test_case['expected']}' in response")
                results.append(False)

        except Exception as e:
            print_error(f"Tool call raised exception: {e}")
            print_info("Exceptions should be handled gracefully and returned as text responses")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(error_test_cases)} error handling tests passed{Colors.ENDC}")

    return results


async def test_parameter_validation(client):
    """Test 7: Parameter validation and type checking."""
    print_section("TEST 7: Parameter Validation", Colors.HEADER)

    validation_tests = [
        {
            'tool': 'search_nodes',
            'args': {'query': 'test', 'limit': 1000},
            'description': 'Large limit parameter (should work but return fewer results)'
        },
        {
            'tool': 'list_nodes_by_type',
            'args': {'node_type': 'file', 'limit': 1},
            'description': 'Minimum limit parameter'
        },
        {
            'tool': 'search_nodes',
            'args': {'query': ''},
            'description': 'Empty search query'
        }
    ]

    results = []
    for i, test_case in enumerate(validation_tests, 1):
        print_subsection(f"Validation Test {i}: {test_case['description']}")
        print(f"LLM action: call_tool('{test_case['tool']}', {json.dumps(test_case['args'])})")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))
            print_success("Parameter validation passed")
            results.append(True)

        except Exception as e:
            print_error(f"Validation test failed: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(validation_tests)} validation tests passed{Colors.ENDC}")

    return results


async def test_json_serialization(client):
    """Test 8: JSON serialization of complex responses."""
    print_section("TEST 8: JSON Serialization & Data Structures", Colors.HEADER)

    print_info("Testing that all responses are properly serializable and structured")

    tools_to_test = ['get_graph_stats', 'search_nodes', 'list_nodes_by_type']
    results = []

    for tool_name in tools_to_test:
        print_subsection(f"Testing {tool_name}")

        try:
            args = {}
            if tool_name == 'search_nodes':
                args = {'query': 'class', 'limit': 2}
            elif tool_name == 'list_nodes_by_type':
                args = {'node_type': 'file', 'limit': 2}

            result = await client.call_tool(tool_name, args)
            response_text = result.data if hasattr(result, "data") else str(result)

            # Check that response is text (not binary or malformed)
            if isinstance(response_text, str):
                print_success(f"Response is properly formatted text ({len(response_text)} characters)")
                results.append(True)
            else:
                print_error(f"Response is not text: {type(response_text)}")
                results.append(False)

        except Exception as e:
            print_error(f"Serialization test failed for {tool_name}: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(tools_to_test)} serialization tests passed{Colors.ENDC}")

    return results


async def test_get_neighbors(client):
    """Test 9: get_neighbors tool - finding directly connected nodes."""
    print_section("TEST 9: Get Neighbors Tool", Colors.HEADER)

    print_info("Testing get_neighbors() - discovers nodes directly connected to a target node")

    # First, search for a node to test with
    print_subsection("Step 1: Find a node to test")
    try:
        search_result = await client.call_tool("search_nodes", {
            "query": "file",
            "limit": 1
        })
        print(search_result)
        search_text = search_result.data if hasattr(search_result, "data") else str(search_result)

        # Extract node ID from search results
        node_id = None
        # Try to extract from header line like: ====================train.py_5===================
        for line in search_text.split('\n'):
            line = line.strip()
            if line.startswith('=') and line.endswith('=') and len(line) > 10:
                # Remove '=' and extract the middle part
                parts = line.strip('=').strip()
                if parts:
                    node_id = parts
                    break
        # Fallback: try old bullet-list style
        if not node_id and "- " in search_text:
            for line in search_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        node_id = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if not node_id:
            print_error("Could not find a valid node to test with")
            return False

        print_success(f"Found test node: {node_id}")

        # Test get_neighbors
        print_subsection("Step 2: Get neighbors of the node")
        print(f"LLM action: call_tool('get_neighbors', {{'node_id': '{node_id}'}})")

        response_result = await client.call_tool("get_neighbors", {
            "node_id": node_id
        })
        response_text = response_result.data if hasattr(response_result, "data") else str(response_result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        # Validate response
        checks = [
            ("Neighbors of" in response_text or "No neighbors" in response_text, "Response contains neighbor information"),
            ("Relation:" in response_text or "No neighbors" in response_text, "Response shows relationships"),
        ]

        print_subsection("Response Validation:")
        all_passed = True
        for check, description in checks:
            if check:
                print_success(description)
            else:
                print_error(description)
                all_passed = False

        # Test with non-existent node
        print_subsection("Step 3: Test with non-existent node")
        error_result = await client.call_tool("get_neighbors", {
            "node_id": "nonexistent_node_xyz"
        })
        error_text = error_result.data if hasattr(error_result, "data") else str(error_result)

        if "not found" in error_text.lower():
            print_success("Error handling works correctly")
        else:
            print_error("Error handling failed")
            all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"get_neighbors test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_entity_tools(client):
    """Test 10: Entity-related tools (go_to_definition, find_usages, list_all_entities)."""
    print_section("TEST 10: Entity Navigation Tools", Colors.HEADER)

    print_info("Testing entity discovery and navigation tools")

    results = []

    # Test 1: list_all_entities
    print_subsection("Test 10.1: list_all_entities")
    print("LLM action: call_tool('list_all_entities', {'limit': 10})")

    try:
        result = await client.call_tool("list_all_entities", {"limit": 10})
        response_text = result.data if hasattr(result, "data") else str(result)

        print("\nServer Response:")
        print(response_text[:400] + ("..." if len(response_text) > 400 else ""))

        # Extract an entity name for testing other tools
        entity_name = None
        if "All Entities" in response_text and not "No entities" in response_text:
            lines = response_text.split('\n')
            for line in lines:
                # Look for pattern like "1. entity_name"
                if line.strip() and line.strip()[0].isdigit() and '.' in line:
                    try:
                        entity_name = line.split('.', 1)[1].strip()
                        break
                    except:
                        pass

        if "All Entities" in response_text or "No entities" in response_text:
            print_success("list_all_entities works correctly")
            results.append(True)
        else:
            print_error("Unexpected response format")
            results.append(False)

    except Exception as e:
        print_error(f"list_all_entities failed: {e}")
        results.append(False)
        entity_name = None

    # Test 2: go_to_definition
    print_subsection("Test 10.2: go_to_definition")

    if entity_name:
        print(f"LLM action: call_tool('go_to_definition', {{'entity_name': '{entity_name}'}})")

        try:
            result = await client.call_tool("go_to_definition", {
                "entity_name": entity_name
            })
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:400] + ("..." if len(response_text) > 400 else ""))

            if "Definition" in response_text or "not found" in response_text or "no declarations" in response_text:
                print_success("go_to_definition works correctly")
                results.append(True)
            else:
                print_error("Unexpected response format")
                results.append(False)

        except Exception as e:
            print_error(f"go_to_definition failed: {e}")
            results.append(False)
    else:
        print_info("Skipping go_to_definition test - no entity found")
        results.append(True)  # Don't fail if no entities exist

    # Test 3: find_usages
    print_subsection("Test 10.3: find_usages")

    if entity_name:
        print(f"LLM action: call_tool('find_usages', {{'entity_name': '{entity_name}', 'limit': 5}})")

        try:
            result = await client.call_tool("find_usages", {
                "entity_name": entity_name,
                "limit": 5
            })
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:400] + ("..." if len(response_text) > 400 else ""))

            if "Usages of" in response_text or "not found" in response_text or "no usages" in response_text:
                print_success("find_usages works correctly")
                results.append(True)
            else:
                print_error("Unexpected response format")
                results.append(False)

        except Exception as e:
            print_error(f"find_usages failed: {e}")
            results.append(False)
    else:
        print_info("Skipping find_usages test - no entity found")
        results.append(True)

    # Test 4: Error handling - non-existent entity
    print_subsection("Test 10.4: Error handling for non-existent entity")

    try:
        result = await client.call_tool("go_to_definition", {
            "entity_name": "nonexistent_entity_xyz_123"
        })
        response_text = result.data if hasattr(result, "data") else str(result)

        if "not found" in response_text.lower():
            print_success("Error handling works for non-existent entities")
            results.append(True)
        else:
            print_error("Error handling failed")
            results.append(False)

    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(results)} entity tool tests passed{Colors.ENDC}")

    return results


async def test_file_and_chunk_tools(client):
    """Test 11: File and chunk navigation tools (get_file_structure, get_related_chunks)."""
    print_section("TEST 11: File & Chunk Navigation Tools", Colors.HEADER)

    print_info("Testing file structure and chunk relationship tools")

    results = []

    # First, find a file node
    print_subsection("Step 1: Find a file node")
    try:
        file_result = await client.call_tool("list_nodes_by_type", {
            "node_type": "file",
            "limit": 1
        })
        file_text = file_result.data if hasattr(file_result, "data") else str(file_result)

        # Extract file path
        file_path = None
        if "- " in file_text and not "No nodes found" in file_text:
            for line in file_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        file_path = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if not file_path:
            print_info("No file nodes found - creating simpler test")
            results.append(True)  # Don't fail if graph has no files
            file_path = None
        else:
            print_success(f"Found test file: {file_path}")

    except Exception as e:
        print_error(f"Failed to find file node: {e}")
        file_path = None

    # Test 1: get_file_structure
    print_subsection("Test 11.1: get_file_structure")

    if file_path:
        print(f"LLM action: call_tool('get_file_structure', {{'file_path': '{file_path}'}})")

        try:
            result = await client.call_tool("get_file_structure", {
                "file_path": file_path
            })
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

            checks = [
                ("File Structure:" in response_text or "not found" in response_text, "Response contains file structure"),
                ("Path:" in response_text or "not found" in response_text, "Response shows file path"),
            ]

            all_passed = True
            for check, description in checks:
                if check:
                    print_success(description)
                    all_passed = all_passed and True
                else:
                    print_error(description)
                    all_passed = False

            results.append(all_passed)

        except Exception as e:
            print_error(f"get_file_structure failed: {e}")
            results.append(False)
    else:
        print_info("Skipping get_file_structure - no file found")
        results.append(True)

    # Test 2: Find a chunk node for testing get_related_chunks
    print_subsection("Test 11.2: Find a chunk node")

    chunk_id = None
    try:
        chunk_result = await client.call_tool("list_nodes_by_type", {
            "node_type": "chunk",
            "limit": 1
        })
        chunk_text = chunk_result.data if hasattr(chunk_result, "data") else str(chunk_result)

        if "- " in chunk_text and not "No nodes found" in chunk_text:
            for line in chunk_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        chunk_id = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if chunk_id:
            print_success(f"Found test chunk: {chunk_id}")
        else:
            print_info("No chunk nodes found")

    except Exception as e:
        print_error(f"Failed to find chunk node: {e}")

    # Test 3: get_related_chunks
    print_subsection("Test 11.3: get_related_chunks")

    if chunk_id:
        print(f"LLM action: call_tool('get_related_chunks', {{'chunk_id': '{chunk_id}', 'relation_type': 'calls'}})")

        try:
            result = await client.call_tool("get_related_chunks", {
                "chunk_id": chunk_id,
                "relation_type": "calls"
            })
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:400] + ("..." if len(response_text) > 400 else ""))

            if "Chunks related to" in response_text or "No chunks found" in response_text or "not found" in response_text:
                print_success("get_related_chunks works correctly")
                results.append(True)
            else:
                print_error("Unexpected response format")
                results.append(False)

        except Exception as e:
            print_error(f"get_related_chunks failed: {e}")
            results.append(False)
    else:
        print_info("Skipping get_related_chunks - no chunk found")
        results.append(True)

    # Test 4: Error handling
    print_subsection("Test 11.4: Error handling for invalid inputs")

    error_tests = [
        ("get_file_structure", {"file_path": "/nonexistent/path/to/file.py"}),
        ("get_related_chunks", {"chunk_id": "nonexistent_chunk_xyz", "relation_type": "calls"}),
    ]

    for tool_name, args in error_tests:
        try:
            result = await client.call_tool(tool_name, args)
            response_text = result.data if hasattr(result, "data") else str(result)

            if "not found" in response_text.lower():
                print_success(f"{tool_name} handles errors correctly")
                results.append(True)
            else:
                print_error(f"{tool_name} error handling unclear")
                results.append(False)

        except Exception as e:
            print_error(f"{tool_name} error test failed: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(results)} file/chunk tool tests passed{Colors.ENDC}")

    return results


async def test_diff_chunks(client):
    """Test 12: diff_chunks tool - comparing two chunks."""
    print_section("TEST 12: Diff Chunks Tool", Colors.HEADER)

    print_info("Testing diff_chunks() - comparing content between two chunks")

    # First, find two chunks to compare
    print_subsection("Step 1: Find chunks to compare")
    try:
        chunk_result = await client.call_tool("list_nodes_by_type", {
            "node_type": "chunk",
            "limit": 2
        })
        chunk_text = chunk_result.data if hasattr(chunk_result, "data") else str(chunk_result)

        # Extract chunk IDs
        chunk_ids = []
        if "- " in chunk_text and not "No nodes found" in chunk_text:
            for line in chunk_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        chunk_id = line.split('-')[1].split(':')[0].strip()
                        chunk_ids.append(chunk_id)
                        if len(chunk_ids) >= 2:
                            break
                    except:
                        pass

        if len(chunk_ids) < 2:
            print_info("Not enough chunks found - testing with error handling")
            # Test error handling instead
            error_result = await client.call_tool("diff_chunks", {
                "node_id_1": "nonexistent_chunk_1",
                "node_id_2": "nonexistent_chunk_2"
            })
            error_text = error_result.data if hasattr(error_result, "data") else str(error_result)
            if "not found" in error_text.lower():
                print_success("Error handling works correctly")
                return True
            else:
                print_error("Error handling failed")
                return False

        print_success(f"Found chunks: {chunk_ids[0]} and {chunk_ids[1]}")

        # Test diff_chunks
        print_subsection("Step 2: Compare the chunks")
        print(f"LLM action: call_tool('diff_chunks', {{'node_id_1': '{chunk_ids[0]}', 'node_id_2': '{chunk_ids[1]}'}})")

        response_result = await client.call_tool("diff_chunks", {
            "node_id_1": chunk_ids[0],
            "node_id_2": chunk_ids[1]
        })
        response_text = response_result.data if hasattr(response_result, "data") else str(response_result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        # Validate response
        checks = [
            ("---" in response_text or "No differences" in response_text or "@@" in response_text,
             "Response contains diff output or indicates no differences"),
        ]

        print_subsection("Response Validation:")
        all_passed = True
        for check, description in checks:
            if check:
                print_success(description)
            else:
                print_error(description)
                all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"diff_chunks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_print_tree(client):
    """Test 13: print_tree tool - showing tree view of repository."""
    print_section("TEST 13: Print Tree Tool", Colors.HEADER)

    print_info("Testing print_tree() - displays hierarchical tree view")

    results = []

    # Test 1: Default tree (no root specified)
    print_subsection("Test 13.1: Default tree view")
    print("LLM action: call_tool('print_tree', {'max_depth': 2})")

    try:
        result = await client.call_tool("print_tree", {"max_depth": 2})
        response_text = result.data if hasattr(result, "data") else str(result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        if "- " in response_text and "(" in response_text:
            print_success("Tree view generated successfully")
            results.append(True)
        else:
            print_error("Unexpected tree format")
            results.append(False)

    except Exception as e:
        print_error(f"print_tree (default) failed: {e}")
        results.append(False)

    # Test 2: Tree from specific root
    print_subsection("Test 13.2: Tree from specific node")

    try:
        # Find a file or directory node
        file_result = await client.call_tool("list_nodes_by_type", {
            "node_type": "file",
            "limit": 1
        })
        file_text = file_result.data if hasattr(file_result, "data") else str(file_result)

        file_id = None
        if "- " in file_text and not "No nodes found" in file_text:
            for line in file_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        file_id = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if file_id:
            print(f"LLM action: call_tool('print_tree', {{'root_id': '{file_id}', 'max_depth': 2}})")
            result = await client.call_tool("print_tree", {
                "root_id": file_id,
                "max_depth": 2
            })
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))

            if "- " in response_text:
                print_success("Tree from specific root works correctly")
                results.append(True)
            else:
                print_error("Unexpected tree format")
                results.append(False)
        else:
            print_info("No file node found - skipping test")
            results.append(True)

    except Exception as e:
        print_error(f"print_tree (specific root) failed: {e}")
        results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(results)} print_tree tests passed{Colors.ENDC}")

    return results


async def test_entity_nodes(client):
    """Test 13.5: Entity Nodes - Testing EntityNode creation and structure."""
    print_section("TEST 13.5: Entity Nodes in Graph", Colors.HEADER)

    print_info("Testing EntityNode creation, structure, and relationships")

    results = []

    # Test 1: Search for entity nodes by type
    print_subsection("Test 13.5.1: Search for entity nodes")
    print("LLM action: call_tool('search_by_type_and_name', {'node_type': 'entity', 'name_query': '', 'limit': 5})")

    try:
        result = await client.call_tool("search_by_type_and_name", {
            "node_type": "entity",
            "name_query": "",
            "limit": 5
        })
        response_text = result.data if hasattr(result, "data") else str(result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        # Extract entity node ID for further testing
        entity_node_id = None
        if hasattr(result, "structured_content") and result.structured_content:
            nodes = result.structured_content.get("nodes", [])
            if nodes:
                entity_node_id = nodes[0].get("node_id")
        
        # Fallback: parse from text
        if not entity_node_id and "entity" in response_text.lower():
            lines = response_text.split('\n')
            for line in lines:
                if line.strip().startswith('=') and line.strip().endswith('='):
                    entity_node_id = line.strip('= ')
                    break

        if "entity" in response_text.lower() or "No nodes found" in response_text:
            print_success("Entity node search works correctly")
            results.append(True)
        else:
            print_error("Unexpected response format")
            results.append(False)

    except Exception as e:
        print_error(f"Entity node search failed: {e}")
        results.append(False)
        entity_node_id = None

    # Test 2: Get entity node info
    print_subsection("Test 13.5.2: Get entity node information")

    if entity_node_id:
        print(f"LLM action: call_tool('get_node_info', {{'node_id': '{entity_node_id}'}})")

        try:
            result = await client.call_tool("get_node_info", {
                "node_id": entity_node_id
            })
            
            response_text = result.data if hasattr(result, "data") else str(result)
            
            print("\nServer Response:")
            pretty_print_node_info(result)

            # Validate EntityNode structure
            checks = [
                ("EntityNode" in response_text or "entity" in response_text.lower(),
                 "Node is identified as EntityNode or entity type"),
                (entity_node_id in response_text,
                 "Entity node ID is present"),
            ]

            print("\nValidation Checks:")
            all_passed = True
            for check, description in checks:
                if check:
                    print_success(description)
                else:
                    print_error(description)
                    all_passed = False

            results.append(all_passed)

        except Exception as e:
            print_error(f"get_node_info for entity failed: {e}")
            results.append(False)
    else:
        print_info("Skipping entity node info test - no entity node found")
        results.append(True)

    # Test 3: Check entity node relationships (declares and called_by edges)
    print_subsection("Test 13.5.3: Verify entity node relationships")

    if entity_node_id:
        print(f"LLM action: call_tool('entity_relationships', {{'node_id': '{entity_node_id}'}})")

        try:
            result = await client.call_tool("entity_relationships", {
                "node_id": entity_node_id
            })
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:600] + ("..." if len(response_text) > 600 else ""))

            # Check for expected relationship types
            checks = [
                ("declares" in response_text.lower() or "called_by" in response_text.lower() or "No relationships" in response_text,
                 "Entity node has declares/called_by relationships or no relationships"),
                ("←" in response_text or "→" in response_text or "No relationships" in response_text,
                 "Shows relationship direction"),
            ]

            print("\nValidation Checks:")
            all_passed = True
            for check, description in checks:
                if check:
                    print_success(description)
                else:
                    print_error(description)
                    all_passed = False

            results.append(all_passed)

        except Exception as e:
            print_error(f"entity_relationships test failed: {e}")
            results.append(False)
    else:
        print_info("Skipping entity relationships test - no entity node found")
        results.append(True)

    # Test 4: Verify entity nodes use entity_name as ID
    print_subsection("Test 13.5.4: Verify entity node ID equals entity name")

    if entity_node_id:
        print(f"Testing that entity node ID '{entity_node_id}' represents the entity name")

        try:
            result = await client.call_tool("get_node_info", {
                "node_id": entity_node_id
            })
            
            if hasattr(result, "structured_content") and result.structured_content:
                node_name = result.structured_content.get("name", "")
                node_id = result.structured_content.get("node_id", "")
                
                if node_name == node_id or node_name in node_id or node_id in node_name:
                    print_success(f"Entity node ID matches entity name: {entity_node_id}")
                    results.append(True)
                else:
                    print_error(f"Entity node ID '{node_id}' doesn't match name '{node_name}'")
                    results.append(False)
            else:
                # Fallback to text parsing
                response_text = result.data if hasattr(result, "data") else str(result)
                if entity_node_id in response_text:
                    print_success("Entity node ID is used consistently")
                    results.append(True)
                else:
                    print_error("Could not verify entity node ID")
                    results.append(False)

        except Exception as e:
            print_error(f"Entity node ID verification failed: {e}")
            results.append(False)
    else:
        print_info("Skipping entity node ID test - no entity node found")
        results.append(True)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(results)} entity node tests passed{Colors.ENDC}")

    return all(results)


async def test_entity_relationships(client):
    """Test 14: entity_relationships tool - showing all relationships for an entity."""
    print_section("TEST 14: Entity Relationships Tool", Colors.HEADER)

    print_info("Testing entity_relationships() - shows all edges for a node")

    # First, find a node with relationships
    print_subsection("Step 1: Find a node to test")
    try:
        search_result = await client.call_tool("search_nodes", {
            "query": "function",
            "limit": 1
        })
        search_text = search_result.data if hasattr(search_result, "data") else str(search_result)

        # Extract node ID
        node_id = None
        for line in search_text.split('\n'):
            line = line.strip()
            if line.startswith('=') and line.endswith('=') and len(line) > 10:
                parts = line.strip('=').strip()
                if parts:
                    node_id = parts
                    break

        if not node_id:
            print_info("Could not find node - testing error handling")
            error_result = await client.call_tool("entity_relationships", {
                "node_id": "nonexistent_node_xyz"
            })
            error_text = error_result.data if hasattr(error_result, "data") else str(error_result)
            if "not found" in error_text.lower():
                print_success("Error handling works correctly")
                return True
            else:
                print_error("Error handling failed")
                return False

        print_success(f"Found test node: {node_id}")

        # Test entity_relationships
        print_subsection("Step 2: Get entity relationships")
        print(f"LLM action: call_tool('entity_relationships', {{'node_id': '{node_id}'}})")

        response_result = await client.call_tool("entity_relationships", {
            "node_id": node_id
        })
        response_text = response_result.data if hasattr(response_result, "data") else str(response_result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        # Validate response
        checks = [
            ("Relationships for" in response_text or "No relationships" in response_text,
             "Response contains relationship information"),
            ("←" in response_text or "→" in response_text or "No relationships" in response_text,
             "Response shows incoming/outgoing relationships"),
        ]

        print_subsection("Response Validation:")
        all_passed = True
        for check, description in checks:
            if check:
                print_success(description)
            else:
                print_error(description)
                all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"entity_relationships test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_by_type_and_name(client):
    """Test 15: search_by_type_and_name tool - searching with type and name filters."""
    print_section("TEST 15: Search By Type And Name Tool", Colors.HEADER)

    print_info("Testing search_by_type_and_name() - filtered search by node type and name")

    test_cases = [
        {
            'node_type': 'file',
            'name_query': 'test',
            'limit': 5,
            'description': 'Search for files with "test" in name'
        },
        {
            'node_type': 'function',
            'name_query': 'main',
            'limit': 3,
            'description': 'Search for functions with "main" in name'
        },
        {
            'node_type': 'class',
            'name_query': 'controller',
            'limit': 5,
            'description': 'Search for classes with "controller" in name'
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        # Extract only the tool parameters (exclude 'description')
        tool_params = {k: v for k, v in test_case.items() if k != 'description'}
        print(f"LLM action: call_tool('search_by_type_and_name', {json.dumps(tool_params)})")

        try:
            result = await client.call_tool("search_by_type_and_name", tool_params)
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            if len(response_text) > 300:
                print(response_text[:300] + "\n... (truncated)")
            else:
                print(response_text)

            # Check for valid response (either matches or no matches)
            if "Matches for" in response_text or "No matches" in response_text:
                print_success(f"search_by_type_and_name executed successfully")
                results.append(True)
            else:
                print_error(f"Unexpected response format")
                results.append(False)

        except Exception as e:
            print_error(f"search_by_type_and_name failed: {e}")
            results.append(False)

    # Test error handling
    print_subsection("Test Case 4: Error handling")
    try:
        result = await client.call_tool("search_by_type_and_name", {
            "node_type": "nonexistent_type",
            "name_query": "xyz",
            "limit": 5
        })
        response_text = result.data if hasattr(result, "data") else str(result)

        if "No matches" in response_text:
            print_success("Error handling works correctly for invalid type")
            results.append(True)
        else:
            print_error("Error handling unclear")
            results.append(False)
    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(results)} search_by_type_and_name tests passed{Colors.ENDC}")

    return results


async def test_get_chunk_context(client):
    """Test 16: get_chunk_context tool - getting surrounding chunks."""
    print_section("TEST 16: Get Chunk Context Tool", Colors.HEADER)

    print_info("Testing get_chunk_context() - retrieves previous and next chunks")

    # Find a chunk to test
    print_subsection("Step 1: Find a chunk to test")
    try:
        chunk_result = await client.call_tool("list_nodes_by_type", {
            "node_type": "chunk",
            "limit": 1
        })
        chunk_text = chunk_result.data if hasattr(chunk_result, "data") else str(chunk_result)

        chunk_id = None
        if "- " in chunk_text and not "No nodes found" in chunk_text:
            for line in chunk_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        chunk_id = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if not chunk_id:
            print_info("No chunks found - testing error handling")
            error_result = await client.call_tool("get_chunk_context", {
                "node_id": "nonexistent_chunk_xyz"
            })
            error_text = error_result.data if hasattr(error_result, "data") else str(error_result)
            if "not found" in error_text.lower():
                print_success("Error handling works correctly")
                return True
            else:
                print_error("Error handling failed")
                return False

        print_success(f"Found test chunk: {chunk_id}")

        # Test get_chunk_context
        print_subsection("Step 2: Get chunk context")
        print(f"LLM action: call_tool('get_chunk_context', {{'node_id': '{chunk_id}'}})")

        response_result = await client.call_tool("get_chunk_context", {
            "node_id": chunk_id
        })
        response_text = response_result.data if hasattr(response_result, "data") else str(response_result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        # Validate response
        checks = [
            ("Context for chunk" in response_text, "Response contains context information"),
            ("Previous chunk" in response_text or "No previous chunk" in response_text,
             "Response mentions previous chunk status"),
            ("Next chunk" in response_text or "No next chunk" in response_text,
             "Response mentions next chunk status"),
        ]

        print_subsection("Response Validation:")
        all_passed = True
        for check, description in checks:
            if check:
                print_success(description)
            else:
                print_error(description)
                all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"get_chunk_context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_get_file_stats(client):
    """Test 17: get_file_stats tool - getting file statistics."""
    print_section("TEST 17: Get File Stats Tool", Colors.HEADER)

    print_info("Testing get_file_stats() - retrieves statistics for a file or directory")

    # Find a file to test
    print_subsection("Step 1: Find a file to test")
    try:
        file_result = await client.call_tool("list_nodes_by_type", {
            "node_type": "file",
            "limit": 1
        })
        file_text = file_result.data if hasattr(file_result, "data") else str(file_result)

        file_path = None
        if "- " in file_text and not "No nodes found" in file_text:
            for line in file_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        file_path = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if not file_path:
            print_info("No files found - testing error handling")
            error_result = await client.call_tool("get_file_stats", {
                "path": "/nonexistent/file/path.py"
            })
            error_text = error_result.data if hasattr(error_result, "data") else str(error_result)
            if "not found" in error_text.lower() or "No nodes found" in error_text:
                print_success("Error handling works correctly")
                return True
            else:
                print_error("Error handling failed")
                return False

        print_success(f"Found test file: {file_path}")

        # Test get_file_stats
        print_subsection("Step 2: Get file statistics")
        print(f"LLM action: call_tool('get_file_stats', {{'path': '{file_path}'}})")

        response_result = await client.call_tool("get_file_stats", {
            "path": file_path
        })
        response_text = response_result.data if hasattr(response_result, "data") else str(response_result)

        print("\nServer Response:")
        print(response_text[:500] + ("..." if len(response_text) > 500 else ""))

        # Validate response
        checks = [
            ("Statistics for" in response_text, "Response contains statistics information"),
            ("Lines:" in response_text or "Declared entities:" in response_text,
             "Response shows file metrics"),
        ]

        print_subsection("Response Validation:")
        all_passed = True
        for check, description in checks:
            if check:
                print_success(description)
            else:
                print_error(description)
                all_passed = False

        # Test with non-existent path
        print_subsection("Step 3: Test error handling")
        error_result = await client.call_tool("get_file_stats", {
            "path": "/definitely/nonexistent/path/file.py"
        })
        error_text = error_result.data if hasattr(error_result, "data") else str(error_result)

        if "not found" in error_text.lower() or "No nodes found" in error_text:
            print_success("Error handling works correctly")
        else:
            print_error("Error handling unclear")
            all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"get_file_stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_file_names_by_regex(client):
    """Test: search_file_names_by_regex tool - searching file names with regex."""
    print_section("TEST: Search File Names By Regex Tool", Colors.HEADER)

    print_info("Testing search_file_names_by_regex() - finds files matching a regex pattern")

    test_cases = [
        {
            'pattern': r'\.py$',
            'description': 'Find all Python files (ending with .py)',
            'expect_match': True
        },
        {
            'pattern': r'nonexistent_file_pattern_xyz$',
            'description': 'Pattern that should not match any file',
            'expect_match': False
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        print(f"LLM action: call_tool('search_file_names_by_regex', {{'pattern': '{test_case['pattern']}'}})")

        try:
            result = await client.call_tool("search_file_names_by_regex", {"pattern": test_case['pattern']})
            response_text = result.data if hasattr(result, "data") else str(result)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))

            if test_case['expect_match']:
                if "Files matching pattern" in response_text and "- " in response_text:
                    print_success("Files matched as expected")
                    results.append(True)
                else:
                    print_error("Expected matches but none found")
                    results.append(False)
            else:
                if "No file names matched" in response_text:
                    print_success("No matches as expected")
                    results.append(True)
                else:
                    print_error("Expected no matches but found some")
                    results.append(False)
        except Exception as e:
            print_error(f"search_file_names_by_regex failed: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(test_cases)} search_file_names_by_regex tests passed{Colors.ENDC}")

    return all(results)


async def test_advanced_tool_combinations(client):
    """Test 18: Advanced tool combinations - realistic LLM workflows."""
    print_section("TEST 18: Advanced Tool Combinations", Colors.HEADER)

    print_info("Testing realistic multi-tool workflows that an LLM might use")

    # Workflow 1: "Show me where entity X is defined and used"
    print_subsection("Workflow 1: Complete entity lifecycle")
    print(f"{Colors.BOLD}User: 'Where is [entity] defined and how is it used?'{Colors.ENDC}")

    try:
        entities_result = await client.call_tool("list_all_entities", {"limit": 5})
        entities_text = entities_result.data if hasattr(entities_result, "data") else str(entities_result)

        entity_name = None
        if "All Entities" in entities_text and not "No entities" in entities_text:
            lines = entities_text.split('\n')
            for line in lines:
                if line.strip() and line.strip()[0].isdigit() and '.' in line:
                    try:
                        entity_name = line.split('.', 1)[1].strip()
                        break
                    except:
                        pass

        if entity_name:
            print(f"\n{Colors.CYAN}LLM Step 1: Find definition{Colors.ENDC}")
            def_result = await client.call_tool("go_to_definition", {"entity_name": entity_name})
            def_text = def_result.data if hasattr(def_result, "data") else str(def_result)
            print(def_text[:200] + "...")

            print(f"\n{Colors.CYAN}LLM Step 2: Find usages{Colors.ENDC}")
            usage_result = await client.call_tool("find_usages", {"entity_name": entity_name, "limit": 3})
            usage_text = usage_result.data if hasattr(usage_result, "data") else str(usage_result)
            print(usage_text[:200] + "...")

            print_success("Workflow 1 completed successfully")
            workflow1_passed = True
        else:
            print_info("No entities available - skipping workflow 1")
            workflow1_passed = True

    except Exception as e:
        print_error(f"Workflow 1 failed: {e}")
        workflow1_passed = False

    # Workflow 2: "Explore a file's structure and its relationships"
    print_subsection("Workflow 2: Deep dive into a file")
    print(f"{Colors.BOLD}User: 'Tell me about the structure of [file]'{Colors.ENDC}")

    try:
        # Find a file
        file_result = await client.call_tool("list_nodes_by_type", {"node_type": "file", "limit": 1})
        file_text = file_result.data if hasattr(file_result, "data") else str(file_result)

        file_path = None
        if "- " in file_text and not "No nodes found" in file_text:
            for line in file_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        file_path = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if file_path:
            print(f"\n{Colors.CYAN}LLM Step 1: Get file structure{Colors.ENDC}")
            struct_result = await client.call_tool("get_file_structure", {"file_path": file_path})
            struct_text = struct_result.data if hasattr(struct_result, "data") else str(struct_result)
            print(struct_text[:200] + "...")

            print(f"\n{Colors.CYAN}LLM Step 2: Get file neighbors{Colors.ENDC}")
            neighbors_result = await client.call_tool("get_neighbors", {"node_id": file_path})
            neighbors_text = neighbors_result.data if hasattr(neighbors_result, "data") else str(neighbors_result)
            print(neighbors_text[:200] + "...")

            print_success("Workflow 2 completed successfully")
            workflow2_passed = True
        else:
            print_info("No files available - skipping workflow 2")
            workflow2_passed = True

    except Exception as e:
        print_error(f"Workflow 2 failed: {e}")
        workflow2_passed = False

    # Workflow 3: "Find related code through chunk relationships"
    print_subsection("Workflow 3: Navigate through code relationships")
    print(f"{Colors.BOLD}User: 'What does this chunk call?'{Colors.ENDC}")

    try:
        # Find a chunk
        chunk_result = await client.call_tool("list_nodes_by_type", {"node_type": "chunk", "limit": 1})
        chunk_text = chunk_result.data if hasattr(chunk_result, "data") else str(chunk_result)

        chunk_id = None
        if "- " in chunk_text and not "No nodes found" in chunk_text:
            for line in chunk_text.split('\n'):
                if line.strip().startswith('- '):
                    try:
                        chunk_id = line.split('-')[1].split(':')[0].strip()
                        break
                    except:
                        pass

        if chunk_id:
            print(f"\n{Colors.CYAN}LLM Step 1: Get chunk info{Colors.ENDC}")
            info_result = await client.call_tool("get_node_info", {"node_id": chunk_id})
            pretty_print_node_info(info_result)

            print(f"\n{Colors.CYAN}LLM Step 2: Get related chunks{Colors.ENDC}")
            related_result = await client.call_tool("get_related_chunks", {
                "chunk_id": chunk_id,
                "relation_type": "calls"
            })
            related_text = related_result.data if hasattr(related_result, "data") else str(related_result)
            print(related_text[:200] + "...")

            print_success("Workflow 3 completed successfully")
            workflow3_passed = True
        else:
            print_info("No chunks available - skipping workflow 3")
            workflow3_passed = True

    except Exception as e:
        print_error(f"Workflow 3 failed: {e}")
        workflow3_passed = False

    # Summary
    results = [workflow1_passed, workflow2_passed, workflow3_passed]
    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(results)} advanced workflows passed{Colors.ENDC}")

    return all(results)


async def run_comprehensive_tests():
    """Run all comprehensive MCP tests."""

    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("=" * 70)
    print("  COMPREHENSIVE MCP SERVER TEST SUITE")
    print("  Simulating Full LLM Client Experience")
    print("=" * 70)
    print(f"{Colors.ENDC}\n")

    print_info("This test suite validates:")
    print("  • Protocol initialization and handshake")
    print("  • Tool discovery mechanism")
    print("  • Basic and parameterized tool execution")
    print("  • Multi-step tool chaining (LLM reasoning simulation)")
    print("  • Error handling and edge cases")
    print("  • Parameter validation")
    print("  • JSON serialization")
    print("  • Response formatting\n")

    # Get MCP server URL from environment or use default
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:4000')
    mcp_url = base_url + '/mcp'
    print_info(f"Connecting to MCP server at {base_url}...")
    client = Client(mcp_url)

    all_tests_passed = True
    test_results = {}

    try:

        async with client:
            connection = await test_client_connection(client)
            test_results['connection'] = connection

            discovery_result, tools = await test_tool_discovery(client)
            test_results['discovery'] = discovery_result

            basic_result, _ = await test_basic_tool_execution(client)
            test_results['basic_execution'] = basic_result

            param_results = await test_parameterized_tool_execution(client)
            test_results['parameterized_execution'] = all(r[0] for r in param_results)

            test_results['tool_chaining'] = await test_tool_chaining(client)

            error_results = await test_error_handling(client)
            test_results['error_handling'] = all(error_results)

            validation_results = await test_parameter_validation(client)
            test_results['parameter_validation'] = all(validation_results)

            serialization_results = await test_json_serialization(client)
            test_results['json_serialization'] = all(serialization_results)

            test_results['get_neighbors'] = await test_get_neighbors(client)

            entity_results = await test_entity_tools(client)
            test_results['entity_tools'] = all(entity_results)

            file_chunk_results = await test_file_and_chunk_tools(client)
            test_results['file_chunk_tools'] = all(file_chunk_results)

            test_results['diff_chunks'] = await test_diff_chunks(client)

            print_tree_results = await test_print_tree(client)
            test_results['print_tree'] = all(print_tree_results)

            test_results['entity_nodes'] = await test_entity_nodes(client)

            test_results['entity_relationships'] = await test_entity_relationships(client)

            search_by_type_results = await test_search_by_type_and_name(client)
            test_results['search_by_type_and_name'] = all(search_by_type_results)

            test_results['get_chunk_context'] = await test_get_chunk_context(client)

            test_results['get_file_stats'] = await test_get_file_stats(client)

            # Add the new test here
            test_results['search_file_names_by_regex'] = await test_search_file_names_by_regex(client)

            test_results['advanced_workflows'] = await test_advanced_tool_combinations(client)

            # Final summary
            print_section("FINAL TEST SUMMARY", Colors.HEADER)

            print(f"{Colors.BOLD}Test Results:{Colors.ENDC}\n")
            for test_name, passed in test_results.items():
                status = f"{Colors.GREEN}✅ PASSED{Colors.ENDC}" if passed else f"{Colors.RED}❌ FAILED{Colors.ENDC}"
                print(f"  {test_name.replace('_', ' ').title()}: {status}")

            all_tests_passed = all(test_results.values())

            print("\n" + "=" * 70)
            if all_tests_passed:
                print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.ENDC}")
                print(f"{Colors.GREEN}Your MCP server is ready for LLM integration!{Colors.ENDC}")
            else:
                print(f"{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED{Colors.ENDC}")
                print(f"{Colors.YELLOW}Review the failed tests above and fix the issues.{Colors.ENDC}")
            print("=" * 70)

            # LLM Integration Summary
            print_section("LLM Integration Summary", Colors.CYAN)
            print(f"{Colors.BOLD}What the LLM Experience:{Colors.ENDC}\n")
            print("1. 🔍 Discovery: The LLM can see all available tools and their parameters")
            print("2. 🎯 Execution: The LLM can call tools with appropriate arguments")
            print("3. 📊 Responses: The LLM receives well-formatted text responses")
            print("4. 🔗 Chaining: The LLM can chain multiple tool calls for complex queries")
            print("5. 🛡️  Safety: Errors are handled gracefully with informative messages")
            print("\n" + f"{Colors.BOLD}Example LLM Usage:{Colors.ENDC}")
            print(f"{Colors.CYAN}User: 'What controllers exist in this codebase?'{Colors.ENDC}")
            print(f"{Colors.YELLOW}LLM: *calls search_nodes('controller')*{Colors.ENDC}")
            print(f"{Colors.YELLOW}LLM: *calls get_node_info(node_id)*{Colors.ENDC}")
            print(f"{Colors.YELLOW}LLM: *synthesizes response from tool outputs*{Colors.ENDC}")
            print(f"{Colors.GREEN}LLM: 'I found 3 controllers: UserController, ProductController...'{Colors.ENDC}")

    except Exception as e:
        print_error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False

    sys.exit(0 if all_tests_passed else 1)


def main():
    asyncio.run(run_comprehensive_tests())


if __name__ == "__main__":
    main()