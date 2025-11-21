#!/usr/bin/env python3
"""
Test script to verify API endpoint detection functionality.
Tests both API endpoint definitions (backend) and API calls (frontend).
"""

import json
from RepoKnowledgeGraphLib.EntityExtractor import (
    PythonASTEntityExtractor,
    JavaEntityExtractor,
    JavaScriptEntityExtractor
)

def test_python_fastapi():
    """Test FastAPI endpoint detection"""
    code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/users")
def get_users():
    return {"users": []}

@app.post("/api/users")
async def create_user(user: User):
    return user

@app.put("/api/users/{user_id}")
def update_user(user_id: int, user: User):
    return user
'''
    
    extractor = PythonASTEntityExtractor()
    declared, called = extractor.extract_entities(code)
    
    print("\\n=== Python FastAPI Test ===")
    print(f"Declared entities: {len(declared)}")
    for entity in declared:
        if entity['type'] == 'api_endpoint':
            print(f"  API Endpoint: {entity['name']} -> {entity.get('endpoint')} [{entity.get('methods')}]")
    
    # Check if API endpoints were detected
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    assert len(api_endpoints) >= 3, f"Expected at least 3 API endpoints, found {len(api_endpoints)}"
    print(f"✓ Found {len(api_endpoints)} API endpoints")


def test_python_flask():
    """Test Flask endpoint detection"""
    code = '''
from flask import Flask, request

app = Flask(__name__)

@app.route("/api/products", methods=["GET", "POST"])
def products():
    if request.method == "GET":
        return {"products": []}
    return {"created": True}

@app.route("/api/products/<int:id>", methods=["GET"])
def get_product(id):
    return {"id": id}
'''
    
    extractor = PythonASTEntityExtractor()
    declared, called = extractor.extract_entities(code)
    
    print("\\n=== Python Flask Test ===")
    print(f"Declared entities: {len(declared)}")
    for entity in declared:
        if entity['type'] == 'api_endpoint':
            print(f"  API Endpoint: {entity['name']} -> {entity.get('endpoint')} [{entity.get('methods')}]")
    
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    assert len(api_endpoints) >= 2, f"Expected at least 2 API endpoints, found {len(api_endpoints)}"
    print(f"✓ Found {len(api_endpoints)} API endpoints")


def test_javascript_fetch():
    """Test JavaScript fetch API call detection"""
    code = '''
async function getUsers() {
    const response = await fetch('/api/users');
    return response.json();
}

async function getProducts() {
    const response = await fetch('/api/products');
    return response.json();
}
'''
    
    extractor = JavaScriptEntityExtractor()
    declared, called = extractor.extract_entities(code)
    
    print("\\n=== JavaScript Fetch Test ===")
    print(f"Called entities: {len(called)}")
    api_calls = [c for c in called if c.startswith('API:')]
    for call in api_calls:
        print(f"  {call}")
    
    assert len(api_calls) >= 2, f"Expected at least 2 API calls, found {len(api_calls)}"
    print(f"✓ Found {len(api_calls)} API calls")


def test_javascript_axios():
    """Test JavaScript axios API call detection"""
    code = '''
import axios from 'axios';

class UserService {
    async getUsers() {
        return axios.get('/api/users');
    }
    
    async updateUser(id, data) {
        return axios.put(`/api/users/${id}`, data);
    }
    
    deleteUser(id) {
        return axios.delete(`/api/users/${id}`);
    }
}
'''
    
    extractor = JavaScriptEntityExtractor()
    declared, called = extractor.extract_entities(code)
    
    print("\\n=== JavaScript Axios Test ===")
    print(f"Called entities: {len(called)}")
    api_calls = [c for c in called if c.startswith('API:')]
    for call in api_calls:
        print(f"  {call}")
    
    assert len(api_calls) >= 3, f"Expected at least 3 API calls, found {len(api_calls)}"
    print(f"✓ Found {len(api_calls)} API calls")


def test_java_spring():
    """Test Java Spring Boot endpoint detection"""
    code = '''
package com.example.api;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class UserController {
    
    @GetMapping("/users")
    public List<User> getUsers() {
        return userService.findAll();
    }
    
    @PostMapping("/users")
    public User createUser(@RequestBody User user) {
        return userService.save(user);
    }
    
    @PutMapping("/users/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody User user) {
        return userService.update(id, user);
    }
}
'''
    
    extractor = JavaEntityExtractor()
    declared, called = extractor.extract_entities(code)
    
    print("\\n=== Java Spring Boot Test ===")
    print(f"Declared entities: {len(declared)}")
    for entity in declared:
        if entity['type'] == 'api_endpoint':
            print(f"  API Endpoint: {entity['name']} -> {entity.get('endpoint')} [{entity.get('methods')}]")
    
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    assert len(api_endpoints) >= 3, f"Expected at least 3 API endpoints, found {len(api_endpoints)}"
    print(f"✓ Found {len(api_endpoints)} API endpoints")


def test_python_mixed_decorators():
    """Test Python with mixed decorator patterns"""
    code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/items")
def get_items():
    return {"items": []}

@app.post("/api/items")
@some_other_decorator
def create_item(item: dict):
    return item

# Regular function without API decorator
def helper_function():
    return "helper"
'''

    extractor = PythonASTEntityExtractor()
    declared, called = extractor.extract_entities(code)

    print("\\n=== Python Mixed Decorators Test ===")
    print(f"Declared entities: {len(declared)}")
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    regular_functions = [e for e in declared if e['type'] == 'function']

    for entity in api_endpoints:
        print(f"  API Endpoint: {entity['name']} -> {entity.get('endpoint')} [{entity.get('methods')}]")

    assert len(api_endpoints) >= 2, f"Expected at least 2 API endpoints, found {len(api_endpoints)}"
    assert len(regular_functions) >= 1, f"Expected at least 1 regular function, found {len(regular_functions)}"
    print(f"✓ Found {len(api_endpoints)} API endpoints and {len(regular_functions)} regular functions")


def test_java_spring_complex_paths():
    """Test Java Spring Boot with complex path patterns"""
    code = '''
package com.example.api;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1")
public class ProductController {
    
    @GetMapping("/products")
    public List<Product> getAllProducts() {
        return productService.findAll();
    }
    
    @GetMapping("/products/{id}")
    public Product getProduct(@PathVariable Long id) {
        return productService.findById(id);
    }
    
    @DeleteMapping("/products/{id}")
    public void deleteProduct(@PathVariable Long id) {
        productService.delete(id);
    }
}
'''

    extractor = JavaEntityExtractor()
    declared, called = extractor.extract_entities(code)

    print("\\n=== Java Spring Boot Complex Paths Test ===")
    print(f"Declared entities: {len(declared)}")
    for entity in declared:
        if entity['type'] == 'api_endpoint':
            print(f"  API Endpoint: {entity['name']} -> {entity.get('endpoint')} [{entity.get('methods')}]")

    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    assert len(api_endpoints) >= 3, f"Expected at least 3 API endpoints, found {len(api_endpoints)}"

    # Verify base path is combined correctly
    endpoints = [e['endpoint'] for e in api_endpoints]
    assert '/api/v1/products' in endpoints, "Base path should be combined with method path"
    print(f"✓ Found {len(api_endpoints)} API endpoints with correct path combination")


def test_javascript_multiple_libraries():
    """Test detection of various JavaScript HTTP client libraries"""
    code = '''
import axios from 'axios';
import { request } from 'superagent';

// Fetch API
fetch('/api/users');

// Axios
axios.get('/api/products');
axios.post('/api/orders', {data: 'test'});

// Superagent
request.get('/api/items');

// Angular $http (would be in different context but same pattern)
function AngularController($http) {
    $http.get('/api/settings');
}
'''

    extractor = JavaScriptEntityExtractor()
    declared, called = extractor.extract_entities(code)

    print("\\n=== JavaScript Multiple Libraries Test ===")
    print(f"Called entities: {len(called)}")
    api_calls = [c for c in called if c.startswith('API:')]
    for call in api_calls:
        print(f"  {call}")

    assert len(api_calls) >= 4, f"Expected at least 4 API calls, found {len(api_calls)}"
    print(f"✓ Found {len(api_calls)} API calls from different libraries")


def test_template_literal_normalization():
    """Test that JavaScript template literals are normalized correctly"""
    code = '''
const userId = 123;
const version = 'v1';

// Template literals with variables
axios.get(`/api/users/${userId}`);
axios.put(`/api/${version}/users/${userId}`, data);
fetch(`/api/products/${productId}/reviews/${reviewId}`);
'''

    extractor = JavaScriptEntityExtractor()
    declared, called = extractor.extract_entities(code)

    print("\\n=== JavaScript Template Literal Test ===")
    api_calls = [c for c in called if c.startswith('API:')]
    for call in api_calls:
        print(f"  {call}")

    # Check that template literals are normalized to {param}
    assert any('{param}' in call for call in api_calls), "Template literals should be normalized"
    print(f"✓ Template literals normalized correctly")


def test_frontend_backend_matching():
    """Test matching frontend API calls to backend endpoints"""
    backend_code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/users")
def get_users():
    return {"users": []}

@app.post("/api/users")
def create_user(user: dict):
    return user

@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    return {"id": user_id}
'''

    frontend_code = '''
async function fetchUsers() {
    return axios.get('/api/users');
}

async function createUser(data) {
    return axios.post('/api/users', data);
}

async function fetchUser(id) {
    return axios.get(`/api/users/${id}`);
}
'''

    # Extract backend endpoints
    backend_extractor = PythonASTEntityExtractor()
    backend_declared, _ = backend_extractor.extract_entities(backend_code)
    backend_endpoints = [e for e in backend_declared if e['type'] == 'api_endpoint']

    # Extract frontend calls
    frontend_extractor = JavaScriptEntityExtractor()
    _, frontend_called = frontend_extractor.extract_entities(frontend_code)
    api_calls = [c for c in frontend_called if c.startswith('API:')]

    print("\\n=== Frontend-Backend Matching Test ===")
    print(f"Backend endpoints: {len(backend_endpoints)}")
    print(f"Frontend API calls: {len(api_calls)}")

    # Match them
    matches = []
    for call in api_calls:
        parts = call.split(':')
        method, endpoint = parts[1], parts[2]

        for ep in backend_endpoints:
            # Normalize endpoint for comparison (handle template params)
            normalized_call_endpoint = endpoint
            normalized_backend_endpoint = ep['endpoint'].replace('{user_id}', '{param}')

            if normalized_call_endpoint == normalized_backend_endpoint or endpoint == ep['endpoint']:
                if method in ep['methods']:
                    matches.append((call, ep['name']))
                    print(f"  ✓ Match: {call} -> {ep['name']}")

    assert len(matches) >= 2, f"Expected at least 2 matches, found {len(matches)}"
    print(f"✓ Successfully matched {len(matches)} frontend calls to backend endpoints")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\\n=== Edge Cases Test ===")

    # Empty code
    extractor = PythonASTEntityExtractor()
    declared, called = extractor.extract_entities("")
    assert declared == [] and called == [], "Empty code should return empty lists"
    print("  ✓ Empty code handled")

    # Code without API endpoints
    code_no_api = '''
def regular_function():
    return "hello"

class RegularClass:
    def method(self):
        pass
'''
    declared, called = extractor.extract_entities(code_no_api)
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    assert len(api_endpoints) == 0, "Should find no API endpoints"
    print("  ✓ Non-API code handled")

    # JavaScript code without API calls
    js_extractor = JavaScriptEntityExtractor()
    js_code_no_api = '''
function add(a, b) {
    return a + b;
}
'''
    _, called = js_extractor.extract_entities(js_code_no_api)
    api_calls = [c for c in called if c.startswith('API:')]
    assert len(api_calls) == 0, "Should find no API calls"
    print("  ✓ Non-API JavaScript code handled")

    print("✓ All edge cases handled correctly")


def test_api_endpoint_structure():
    """Test that API endpoint entities have correct structure"""
    code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/test")
def test_endpoint():
    return {}
'''

    extractor = PythonASTEntityExtractor()
    declared, _ = extractor.extract_entities(code)

    print("\\n=== API Endpoint Structure Test ===")

    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    assert len(api_endpoints) > 0, "Should find at least one endpoint"

    endpoint = api_endpoints[0]

    # Verify required fields
    assert 'name' in endpoint, "Endpoint should have 'name' field"
    assert 'type' in endpoint, "Endpoint should have 'type' field"
    assert 'endpoint' in endpoint, "Endpoint should have 'endpoint' field"
    assert 'methods' in endpoint, "Endpoint should have 'methods' field"

    assert endpoint['type'] == 'api_endpoint', "Type should be 'api_endpoint'"
    assert isinstance(endpoint['methods'], list), "Methods should be a list"
    assert len(endpoint['methods']) > 0, "Methods list should not be empty"

    print(f"  Endpoint structure: {endpoint}")
    print("✓ API endpoint structure is correct")



def test_api_endpoint_api():
    """Test FastAPI streaming endpoint detection with complex code structure"""
    code = '''
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from StreamingModel import StreamingModel

os.environ['MODEL_NAME'] = 'mistralai/Mistral-7B-v0.3'

class QueryRequest(BaseModel):
    query: str
    max_new_tokens: int

app = FastAPI()
streaming_model = StreamingModel()

@app.post('/query-stream/')
async def stream(request: QueryRequest):
    # Assuming your chat_model.query function can accept max_new_tokens as an argument
    return StreamingResponse(
        streaming_model.query(request.query, max_new_tokens=request.max_new_tokens),
        media_type='text/event-stream'
    )
'''

    extractor = PythonASTEntityExtractor()
    declared, called = extractor.extract_entities(code)

    print("\\n=== Python FastAPI Streaming Endpoint Test ===")
    print(f"Declared entities: {len(declared)}")

    # Find all API endpoints
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    for entity in api_endpoints:
        print(f"  API Endpoint: {entity['name']} -> {entity.get('endpoint')} [{entity.get('methods')}]")

    # Find the QueryRequest class
    classes = [e for e in declared if e['type'] == 'class']
    print(f"  Classes found: {[c['name'] for c in classes]}")

    # Verify the endpoint was detected
    assert len(api_endpoints) >= 1, f"Expected at least 1 API endpoint, found {len(api_endpoints)}"

    # Verify the endpoint details
    stream_endpoint = [e for e in api_endpoints if e['name'] == 'stream']
    assert len(stream_endpoint) == 1, "Should find the 'stream' endpoint"
    assert stream_endpoint[0]['endpoint'] == '/query-stream/', f"Endpoint path should be '/query-stream/', got {stream_endpoint[0]['endpoint']}"
    assert 'POST' in stream_endpoint[0]['methods'], f"Should have POST method, got {stream_endpoint[0]['methods']}"

    print(f"✓ Found streaming endpoint: {stream_endpoint[0]['name']} -> {stream_endpoint[0]['endpoint']}")
    print(f"✓ Correctly detected POST method for streaming endpoint")


# Test functions will be auto-discovered and run by pytest

