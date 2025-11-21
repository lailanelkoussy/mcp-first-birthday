#!/usr/bin/env python3
"""
Test script to verify Rust API endpoint detection functionality.
Tests both Actix-web and Rocket framework patterns.
"""

from RepoKnowledgeGraphLib.EntityExtractor import RustEntityExtractor


def test_actix_web_endpoints():
    """Test Actix-web endpoint detection with route macros"""
    rust_code = '''
use actix_web::{get, post, put, delete, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// Get all users
#[get("/api/users")]
async fn get_users() -> impl Responder {
    HttpResponse::Ok().json(vec![])
}

/// Create a new user
#[post("/api/users")]
async fn create_user(user: web::Json<User>) -> impl Responder {
    HttpResponse::Created().json(user.into_inner())
}

/// Get user by ID
#[get("/api/users/{id}")]
async fn get_user(id: web::Path<u64>) -> impl Responder {
    HttpResponse::Ok().json(User {
        id: *id,
        name: "Test".to_string(),
        email: "test@example.com".to_string(),
    })
}

/// Update user
#[put("/api/users/{id}")]
async fn update_user(id: web::Path<u64>, user: web::Json<User>) -> impl Responder {
    HttpResponse::Ok().json(user.into_inner())
}

/// Delete user
#[delete("/api/users/{id}")]
async fn delete_user(id: web::Path<u64>) -> impl Responder {
    HttpResponse::NoContent().finish()
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(get_users)
       .service(create_user)
       .service(get_user)
       .service(update_user)
       .service(delete_user);
}
'''
    
    extractor = RustEntityExtractor()
    declared, called = extractor.extract_entities(rust_code)
    
    print("\n=== Rust Actix-web Test ===")
    print(f"Declared entities: {len(declared)}")
    
    # Find API endpoints
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    
    print(f"\nFound {len(api_endpoints)} API endpoints:")
    for endpoint in api_endpoints:
        methods = endpoint.get('methods', [])
        path = endpoint.get('endpoint', 'unknown')
        name = endpoint.get('name', 'unknown')
        print(f"  [{', '.join(methods)}] {path} -> {name}")
    
    # Validate results
    assert len(api_endpoints) >= 5, f"Expected at least 5 API endpoints, found {len(api_endpoints)}"
    
    # Check specific endpoints
    endpoint_paths = [e['endpoint'] for e in api_endpoints]
    assert "/api/users" in endpoint_paths, "Missing /api/users endpoint"
    assert "/api/users/{id}" in endpoint_paths, "Missing /api/users/{id} endpoint"
    
    # Check HTTP methods
    methods = [method for e in api_endpoints for method in e.get('methods', [])]
    assert "GET" in methods, "Missing GET method"
    assert "POST" in methods, "Missing POST method"
    assert "PUT" in methods, "Missing PUT method"
    assert "DELETE" in methods, "Missing DELETE method"
    
    print("✓ All Actix-web endpoint tests passed!")


def test_rocket_endpoints():
    """Test Rocket framework endpoint detection"""
    rust_code = '''
#[macro_use] extern crate rocket;

use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Product {
    pub id: u64,
    pub name: String,
    pub price: f64,
}

#[get("/api/products")]
fn get_products() -> Json<Vec<Product>> {
    Json(vec![])
}

#[post("/api/products", data = "<product>")]
fn create_product(product: Json<Product>) -> Json<Product> {
    product
}

#[get("/api/products/<id>")]
fn get_product(id: u64) -> Option<Json<Product>> {
    Some(Json(Product {
        id,
        name: "Test Product".to_string(),
        price: 99.99,
    }))
}

#[put("/api/products/<id>")]
fn update_product(id: u64, product: Json<Product>) -> Json<Product> {
    product
}

#[delete("/api/products/<id>")]
fn delete_product(id: u64) -> &'static str {
    "Deleted"
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![
            get_products, 
            create_product, 
            get_product, 
            update_product, 
            delete_product
        ])
}
'''
    
    extractor = RustEntityExtractor()
    declared, called = extractor.extract_entities(rust_code)
    
    print("\n=== Rust Rocket Test ===")
    print(f"Declared entities: {len(declared)}")
    
    # Find API endpoints
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    
    print(f"\nFound {len(api_endpoints)} API endpoints:")
    for endpoint in api_endpoints:
        methods = endpoint.get('methods', [])
        path = endpoint.get('endpoint', 'unknown')
        name = endpoint.get('name', 'unknown')
        print(f"  [{', '.join(methods)}] {path} -> {name}")
    
    # Validate results
    assert len(api_endpoints) >= 5, f"Expected at least 5 API endpoints, found {len(api_endpoints)}"
    
    # Check specific endpoints
    endpoint_paths = [e['endpoint'] for e in api_endpoints]
    assert "/api/products" in endpoint_paths, "Missing /api/products endpoint"
    assert "/api/products/<id>" in endpoint_paths, "Missing /api/products/<id> endpoint"
    
    print("✓ All Rocket endpoint tests passed!")


def test_mixed_functions():
    """Test that non-API functions are still properly detected"""
    rust_code = '''
use actix_web::{get, HttpResponse};

// Regular function without API route
fn helper_function(x: i32) -> i32 {
    x * 2
}

// API endpoint
#[get("/api/test")]
async fn test_endpoint() -> HttpResponse {
    HttpResponse::Ok().body("test")
}

// Another regular function
pub fn process_data(data: &str) -> String {
    data.to_uppercase()
}
'''
    
    extractor = RustEntityExtractor()
    declared, called = extractor.extract_entities(rust_code)
    
    print("\n=== Mixed Functions Test ===")
    
    # Find different entity types
    api_endpoints = [e for e in declared if e['type'] == 'api_endpoint']
    functions = [e for e in declared if e['type'] == 'function']
    
    print(f"API endpoints: {len(api_endpoints)}")
    print(f"Regular functions: {len(functions)}")
    
    # Validate
    assert len(api_endpoints) == 1, f"Expected 1 API endpoint, found {len(api_endpoints)}"
    assert len(functions) == 2, f"Expected 2 regular functions, found {len(functions)}"
    
    # Check that the API endpoint is correctly identified
    endpoint = api_endpoints[0]
    assert endpoint['endpoint'] == "/api/test", f"Wrong endpoint path: {endpoint['endpoint']}"
    assert endpoint['methods'] == ['GET'], f"Wrong HTTP method: {endpoint['methods']}"
    
    print("✓ Mixed functions test passed!")


# Test functions will be auto-discovered and run by pytest

