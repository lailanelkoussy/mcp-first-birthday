"""Debug script to test C++ entity extraction"""
import tempfile
import os
from RepoKnowledgeGraphLib.EntityExtractor import CppEntityExtractor
from clang import cindex


def test_cpp_entity_extraction_debug():
    """Debug test for C++ entity extraction with includes and namespaces."""
    # Create temporary files
    tmpdir = tempfile.mkdtemp()

    # Create header file
    header_path = os.path.join(tmpdir, "math", "calculator.h")
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    with open(header_path, 'w') as f:
        f.write("""
namespace math {
    class Calculator {
    public:
        int add(int a, int b);
    };
}
""")

    # Create implementation file
    impl_path = os.path.join(tmpdir, "main.cpp")
    with open(impl_path, 'w') as f:
        f.write("""
#include "math/calculator.h"

using namespace math;

int main() {
    Calculator calc;  // Called as 'Calculator', should resolve to 'math::Calculator'
    return calc.add(1, 2);
}
""")

    # Test extraction
    extractor = CppEntityExtractor()

    print("\n" + "="*80)
    print("TESTING C++ ENTITY EXTRACTION")
    print("="*80)

    print("\n--- Header file (calculator.h) ---")
    with open(header_path, 'r') as f:
        header_code = f.read()
        print(header_code)

    declared_header, called_header = extractor.extract_entities(header_code)
    print(f"\nDeclared entities: {declared_header}")
    print(f"Called entities: {called_header}")

    print("\n--- Implementation file (main.cpp) ---")
    with open(impl_path, 'r') as f:
        impl_code = f.read()
        print(impl_code)

    declared, called = extractor.extract_entities(impl_code)
    print(f"\nDeclared entities: {declared}")
    print(f"Called entities: {called}")

    # Manual check for TYPE_REF in the implementation
    print("\n--- Checking for TYPE_REF nodes manually ---")
    index2 = cindex.Index.create()
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w+", delete=False) as tf:
        tf.write(impl_code)
        tf.flush()
        tf_name = tf.name

    tu3 = index2.parse(tf_name, args=['-std=c++17', '-xc++'])

    def find_var_decls(cursor):
        for c in cursor.get_children():
            if c.kind == cindex.CursorKind.VAR_DECL:
                print(f"\nFound VAR_DECL: '{c.spelling}' (type: '{c.type.spelling}')")
                print("  Children:")
                for child in c.get_children():
                    print(f"    {child.kind}: '{child.spelling}' (type: '{child.type.spelling}')")
                    if child.kind == cindex.CursorKind.TYPE_REF:
                        print(f"      -> TYPE_REF found! This is the type reference we need!")
            find_var_decls(c)

    find_var_decls(tu3.cursor)
    os.unlink(tf_name)

    # Now test with raw clang parsing to see what's happening
    print("\n--- RAW CLANG ANALYSIS ---")
    index = cindex.Index.create()

    # Write code to temp file
    test_file = os.path.join(tmpdir, "test.cpp")
    with open(test_file, 'w') as f:
        f.write("""
#include "math/calculator.h"

using namespace math;

int main() {
    Calculator calc;
    return calc.add(1, 2);
}
""")

    # Parse with include path
    tu = index.parse(
        test_file,
        args=['-std=c++17', '-xc++', f'-I{tmpdir}'],
        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    )

    def walk_and_print(cursor, depth=0):
        indent = "  " * depth
        # Only filter to main file if location exists, otherwise show all
        show = True
        if cursor.location.file:
            # Show nodes from test.cpp only
            if 'test.cpp' not in cursor.location.file.name:
                show = False

        if show:
            print(f"{indent}{cursor.kind}: '{cursor.spelling}' (type: '{cursor.type.spelling}')")
            if cursor.kind == cindex.CursorKind.VAR_DECL:
                print(f"{indent}  -> VAR_DECL Children:")
                for child in cursor.get_children():
                    print(f"{indent}    {child.kind}: '{child.spelling}' (type: '{child.type.spelling}')")

        # Always recurse
        for child in cursor.get_children():
            walk_and_print(child, depth + 1)

    print("\nAST for test.cpp:")
    walk_and_print(tu.cursor)

    # Also test without include path
    print("\n--- PARSING WITHOUT INCLUDE PATH ---")
    tu2 = index.parse(
        test_file,
        args=['-std=c++17', '-xc++'],
        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    )

    print("\nAST for test.cpp (no includes):")
    walk_and_print(tu2.cursor)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)

    # Assertions for pytest
    assert declared_header is not None, "Header declared entities should not be None"
    assert declared is not None, "Implementation declared entities should not be None"

