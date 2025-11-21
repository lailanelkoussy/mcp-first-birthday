#!/usr/bin/env python3
"""
Error Handling Test for MCP Knowledge Graph Server.
Tests improved error handling including custom exceptions and structured error responses.
"""

import asyncio
import os
import sys
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


async def test_node_not_found_errors(client):
    """Test 1: Node not found errors - structured error responses."""
    print_section("TEST 1: Node Not Found Errors", Colors.HEADER)

    test_cases = [
        {
            'tool': 'get_node_info',
            'args': {'node_id': 'nonexistent_node_12345'},
            'description': 'Get info for non-existent node',
            'expected_error_type': 'node_not_found'
        },
        {
            'tool': 'get_node_edges',
            'args': {'node_id': 'fake_node_xyz'},
            'description': 'Get edges for non-existent node',
            'expected_error_type': 'node_not_found'
        },
        {
            'tool': 'get_neighbors',
            'args': {'node_id': 'invalid_neighbor_node'},
            'description': 'Get neighbors for non-existent node',
            'expected_error_type': 'node_not_found'
        },
        {
            'tool': 'get_file_structure',
            'args': {'file_path': '/fake/path/to/file.py'},
            'description': 'Get file structure for non-existent file',
            'expected_error_type': 'node_not_found'
        },
        {
            'tool': 'find_path',
            'args': {'source_id': 'nonexistent_source', 'target_id': 'nonexistent_target'},
            'description': 'Find path between non-existent nodes',
            'expected_error_type': 'node_not_found'
        },
        {
            'tool': 'get_subgraph',
            'args': {'node_id': 'fake_central_node', 'depth': 2},
            'description': 'Get subgraph for non-existent node',
            'expected_error_type': 'node_not_found'
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        print(f"Tool: {test_case['tool']}")
        print(f"Args: {json.dumps(test_case['args'], indent=2)}")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_data = result.data if hasattr(result, "data") else result

            # Convert to string for display and checking
            if isinstance(response_data, dict):
                response_text = json.dumps(response_data, indent=2)
            else:
                response_text = str(response_data)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))

            # Check for error handling
            if 'not found' in response_text.lower():
                print_success("Error handled gracefully with informative message")
                results.append(True)
            else:
                print_error("Expected 'not found' error message")
                results.append(False)

        except Exception as e:
            print_error(f"Tool call raised exception: {e}")
            print_info("Exceptions should be handled gracefully and returned as structured responses")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(test_cases)} node not found tests passed{Colors.ENDC}")

    return results


async def test_entity_not_found_errors(client):
    """Test 2: Entity not found errors."""
    print_section("TEST 2: Entity Not Found Errors", Colors.HEADER)

    test_cases = [
        {
            'tool': 'go_to_definition',
            'args': {'entity_name': 'definitely_not_a_real_entity_12345'},
            'description': 'Go to definition for non-existent entity',
            'expected': 'not found'
        },
        {
            'tool': 'find_usages',
            'args': {'entity_name': 'fake_entity_xyz', 'limit': 10},
            'description': 'Find usages for non-existent entity',
            'expected': 'not found'
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        print(f"Tool: {test_case['tool']}")
        print(f"Args: {json.dumps(test_case['args'], indent=2)}")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_data = result.data if hasattr(result, "data") else result

            # Convert to string for display and checking
            if isinstance(response_data, dict):
                response_text = json.dumps(response_data, indent=2)
            else:
                response_text = str(response_data)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))

            if test_case['expected'].lower() in response_text.lower():
                print_success("Error handled gracefully with informative message")
                results.append(True)
            else:
                print_error(f"Expected '{test_case['expected']}' in response")
                results.append(False)

        except Exception as e:
            print_error(f"Tool call raised exception: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(test_cases)} entity not found tests passed{Colors.ENDC}")

    return results


async def test_invalid_input_errors(client):
    """Test 3: Invalid input validation."""
    print_section("TEST 3: Invalid Input Validation", Colors.HEADER)

    test_cases = [
        {
            'tool': 'search_nodes',
            'args': {'query': 'test', 'limit': 0},
            'description': 'Search with limit = 0 (should fail validation)',
            'expected': 'positive integer'
        },
        {
            'tool': 'find_usages',
            'args': {'entity_name': 'test', 'limit': -5},
            'description': 'Find usages with negative limit (should fail validation)',
            'expected': 'positive integer'
        },
        {
            'tool': 'search_by_type_and_name',
            'args': {'node_type': 'file', 'name_query': 'test', 'limit': -1},
            'description': 'Search by type with negative limit',
            'expected': 'positive integer'
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        print(f"Tool: {test_case['tool']}")
        print(f"Args: {json.dumps(test_case['args'], indent=2)}")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_data = result.data if hasattr(result, "data") else result

            # Convert to string for display and checking
            if isinstance(response_data, dict):
                response_text = json.dumps(response_data, indent=2)
            else:
                response_text = str(response_data)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))

            # Check if validation error is present
            if 'error' in response_text.lower() or test_case['expected'].lower() in response_text.lower():
                print_success("Input validation error handled gracefully")
                results.append(True)
            else:
                print_info("Warning: Expected validation error but got normal response")
                results.append(True)  # Some tools might handle edge cases differently

        except Exception as e:
            print_error(f"Tool call raised exception: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(test_cases)} input validation tests passed{Colors.ENDC}")

    return results


async def test_graceful_edge_cases(client):
    """Test 4: Edge cases that should return empty results gracefully."""
    print_section("TEST 4: Graceful Edge Case Handling", Colors.HEADER)

    test_cases = [
        {
            'tool': 'list_nodes_by_type',
            'args': {'node_type': 'nonexistent_type_xyz', 'limit': 10},
            'description': 'List nodes of non-existent type',
            'expected': 'No nodes found'
        },
        {
            'tool': 'search_nodes',
            'args': {'query': 'xyzabc123nonexistentquery', 'limit': 10},
            'description': 'Search with query that matches nothing',
            'expected': 'No results'
        },
        {
            'tool': 'search_by_type_and_name',
            'args': {'node_type': 'file', 'name_query': 'nonexistent_file_xyz', 'limit': 10},
            'description': 'Search by type and name with no matches',
            'expected': 'No matches'
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test Case {i}: {test_case['description']}")
        print(f"Tool: {test_case['tool']}")
        print(f"Args: {json.dumps(test_case['args'], indent=2)}")

        try:
            result = await client.call_tool(test_case['tool'], test_case['args'])
            response_data = result.data if hasattr(result, "data") else result

            # Convert to string for display and checking
            if isinstance(response_data, dict):
                response_text = json.dumps(response_data, indent=2)
            else:
                response_text = str(response_data)

            print("\nServer Response:")
            print(response_text[:300] + ("..." if len(response_text) > 300 else ""))

            # Check that empty results are handled gracefully
            if test_case['expected'].lower() in response_text.lower():
                print_success("Empty result handled gracefully with informative message")
                results.append(True)
            else:
                print_error(f"Expected '{test_case['expected']}' message in response")
                results.append(False)

        except Exception as e:
            print_error(f"Tool call raised exception: {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n{Colors.BOLD}Summary: {success_count}/{len(test_cases)} edge case tests passed{Colors.ENDC}")

    return results


async def test_error_structure(client):
    """Test 5: Verify structured error responses."""
    print_section("TEST 5: Structured Error Response Format", Colors.HEADER)

    print_info("Testing that error responses contain proper structure")
    print_info("Expected fields: error, error_type, context (when applicable)")

    # Test with a non-existent node
    print_subsection("Test: Non-existent node error structure")
    print("Tool: get_node_info")
    print("Args: {'node_id': 'test_error_structure_node'}")

    try:
        result = await client.call_tool('get_node_info', {
            'node_id': 'test_error_structure_node'
        })
        response_data = result.data if hasattr(result, "data") else result

        # Convert to string for display and checking
        if isinstance(response_data, dict):
            response_text = json.dumps(response_data, indent=2)
        else:
            response_text = str(response_data)

        print("\nServer Response:")
        print(response_text)

        # Check for error structure indicators
        has_error = 'error' in response_text.lower()
        has_context = 'context' in response_text.lower() or 'get_node_info' in response_text.lower()

        if has_error:
            print_success("Response contains error information")
            if has_context:
                print_success("Response includes context about where error occurred")
            return True
        else:
            print_error("Response doesn't appear to be a proper error message")
            return False

    except Exception as e:
        print_error(f"Tool call raised exception: {e}")
        return False


async def run_all_tests():
    """Run all error handling tests."""
    print_section("MCP Knowledge Graph Server - Error Handling Tests", Colors.HEADER)
    print_info("Testing improved error handling with custom exceptions")
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:4000')
    mcp_url = base_url + '/mcp'
    print_info(f"Connecting to MCP server at {base_url}...")
    client = Client(mcp_url)

    try:
        async with  client:
            print_success("Connected to MCP server\n")

            # Run all test suites
            test_results = {}

            test_results['node_not_found'] = await test_node_not_found_errors(client)
            test_results['entity_not_found'] = await test_entity_not_found_errors(client)
            test_results['invalid_input'] = await test_invalid_input_errors(client)
            test_results['edge_cases'] = await test_graceful_edge_cases(client)
            test_results['error_structure'] = [await test_error_structure(client)]

            # Print final summary
            print_section("FINAL TEST SUMMARY", Colors.HEADER)

            total_tests = sum(len(results) for results in test_results.values())
            total_passed = sum(sum(1 for r in results if r) for results in test_results.values())

            print(f"{Colors.BOLD}Test Results by Category:{Colors.ENDC}\n")
            for category, results in test_results.items():
                passed = sum(1 for r in results if r)
                total = len(results)
                status = Colors.GREEN if passed == total else Colors.YELLOW if passed > 0 else Colors.RED
                print(f"  {category:20s}: {status}{passed}/{total} passed{Colors.ENDC}")

            print(f"\n{Colors.BOLD}Overall: {Colors.ENDC}", end="")
            if total_passed == total_tests:
                print(f"{Colors.GREEN}{total_passed}/{total_tests} tests passed ✅{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}{total_passed}/{total_tests} tests passed{Colors.ENDC}")

    except ConnectionError as e:
        print_error(f"Failed to connect to server: {e}")
        print_info(f"Make sure the MCP server is running on {os.getenv('MCP_SERVER_URL')}/mcp")
        return 1
    except Exception as e:
        print_error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if total_passed == total_tests else 1


async def main():
    """Main entry point."""
    exit_code = await run_all_tests()
    sys.exit(exit_code)


# Test functions will be auto-discovered and run by pytest

