#!/usr/bin/env python3
"""
Test script to verify that imports are being detected in entity extraction
"""

from RepoKnowledgeGraphLib.EntityExtractor import (
    PythonASTEntityExtractor,
    JavaEntityExtractor,
    JavaScriptEntityExtractor,
    CEntityExtractor,
    CppEntityExtractor,
    RustEntityExtractor,
    HybridEntityExtractor
)

def test_python_imports():
    print("\n=== Testing Python Import Detection ===")
    extractor = PythonASTEntityExtractor()
    
    code = """
import os
import sys
from typing import List, Dict
from pathlib import Path
import numpy as np

def hello():
    print("Hello")
"""
    
    declared, called = extractor.extract_entities(code)
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that imports are detected
    assert "os" in called, "os import should be detected"
    assert "sys" in called, "sys import should be detected"
    assert "typing" in called, "typing import should be detected"
    assert "pathlib" in called, "pathlib import should be detected"
    assert "numpy" in called, "numpy import should be detected"
    
    print("✅ Python import detection: PASSED")


def test_java_imports():
    print("\n=== Testing Java Import Detection ===")
    extractor = JavaEntityExtractor()
    
    code = """
package com.example;

import java.util.List;
import java.util.ArrayList;
import org.springframework.web.bind.annotation.*;

public class MyClass {
    public void myMethod() {
        System.out.println("Hello");
    }
}
"""
    
    declared, called = extractor.extract_entities(code)
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that imports are detected
    assert "java.util.List" in called, "java.util.List import should be detected"
    assert "java.util.ArrayList" in called, "java.util.ArrayList import should be detected"
    # The wildcard import is captured as the base package path without the wildcard
    assert any("org.springframework.web.bind.annotation" in c for c in called), "Spring import should be detected"

    print("✅ Java import detection: PASSED")


def test_javascript_imports():
    print("\n=== Testing JavaScript Import Detection ===")
    extractor = JavaScriptEntityExtractor()
    
    code = """
import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';

function MyComponent() {
    console.log("Hello");
    return "Hello";
}
"""
    
    declared, called = extractor.extract_entities(code)
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that imports are detected
    assert "react" in called, "react import should be detected"
    assert "axios" in called, "axios import should be detected"
    
    print("✅ JavaScript import detection: PASSED")


def test_c_includes():
    print("\n=== Testing C Include Detection ===")
    extractor = CEntityExtractor()
    
    code = """
#include <stdio.h>
#include <stdlib.h>
#include "myheader.h"

int main() {
    printf("Hello\\n");
    return 0;
}
"""
    
    declared, called = extractor.extract_entities(code)
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that includes are detected
    assert "stdio.h" in called or any("stdio.h" in c for c in called), "stdio.h include should be detected"
    assert "stdlib.h" in called or any("stdlib.h" in c for c in called), "stdlib.h include should be detected"
    assert "myheader.h" in called or any("myheader.h" in c for c in called), "myheader.h include should be detected"
    
    print("✅ C include detection: PASSED")


def test_cpp_includes():
    print("\n=== Testing C++ Include Detection ===")
    extractor = CppEntityExtractor()
    
    code = """
#include <iostream>
#include <vector>
#include <string>
#include "myclass.hpp"

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
"""
    
    declared, called = extractor.extract_entities(code)
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that includes are detected
    assert "iostream" in called or any("iostream" in c for c in called), "iostream include should be detected"
    assert "vector" in called or any("vector" in c for c in called), "vector include should be detected"
    assert "string" in called or any("string" in c for c in called), "string include should be detected"
    assert "myclass.hpp" in called or any("myclass.hpp" in c for c in called), "myclass.hpp include should be detected"
    
    print("✅ C++ include detection: PASSED")


def test_rust_use_declarations():
    print("\n=== Testing Rust Use Declaration Detection ===")
    extractor = RustEntityExtractor()
    
    code = """
use std::collections::HashMap;
use std::io::Result;
use actix_web::{web, App, HttpServer};

fn main() {
    println!("Hello");
}
"""
    
    declared, called = extractor.extract_entities(code)
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that use declarations are detected
    assert any("std::collections::HashMap" in c for c in called), "std::collections::HashMap should be detected"
    assert any("std::io::Result" in c for c in called), "std::io::Result should be detected"
    assert any("actix_web" in c for c in called), "actix_web should be detected"
    
    print("✅ Rust use declaration detection: PASSED")


def test_hybrid_extractor():
    print("\n=== Testing Hybrid Extractor with Python ===")
    extractor = HybridEntityExtractor()
    
    code = """
import os
from typing import List

def greet(name: str) -> str:
    return f"Hello {name}"
"""
    
    declared, called = extractor.extract_entities(code, "test.py")
    
    print("Declared entities:", [e['name'] for e in declared])
    print("Called entities:", called)
    
    # Check that imports are detected
    assert "os" in called, "os import should be detected"
    assert "typing" in called, "typing import should be detected"
    
    print("✅ Hybrid extractor import detection: PASSED")


# Test functions will be auto-discovered and run by pytest

