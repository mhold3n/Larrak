import ast
import importlib.util
import os
import re


def get_python_files(directory):
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def check_imports(file_path):
    broken_imports = []
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read(), filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not is_import_valid(alias.name):
                        broken_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and not is_import_valid(node.module):
                    broken_imports.append(node.module)
    except Exception:
        # print(f"Error parsing {file_path}: {e}")
        pass
    return broken_imports


def is_import_valid(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except:
        return False


def check_file_references(file_path, project_root):
    missing_files = []
    try:
        with open(file_path) as f:
            content = f.read()

        # Regex to find strings that look like file paths
        # Look for strings ending in common extensions
        matches = re.findall(
            r'[\'"]([^\'"]+\.(?:py|txt|log|json|csv|yaml|yml|md))[\'"]', content
        )

        for match in matches:
            # Skip common false positives
            if match.startswith("http") or "*" in match or "{" in match:
                continue

            # Check if file exists relative to current file or project root
            # 1. Absolute path check (if it looks absolute)
            if match.startswith("/"):
                if not os.path.exists(match):
                    missing_files.append(match)
                continue

            # 2. Relative to project root
            root_path = os.path.join(project_root, match)

            # 3. Relative to the script itself
            script_dir = os.path.dirname(file_path)
            rel_path = os.path.join(script_dir, match)

            if not os.path.exists(root_path) and not os.path.exists(rel_path):
                missing_files.append(match)

    except Exception:
        pass
    return missing_files


def main():
    project_root = os.getcwd()
    dirs_to_check = ["scripts", "tests"]

    print(f"Analyzing dead code in {dirs_to_check}...")

    for d in dirs_to_check:
        dir_path = os.path.join(project_root, d)
        if not os.path.exists(dir_path):
            continue

        files = get_python_files(dir_path)
        for f in files:
            rel_path = os.path.relpath(f, project_root)

            # Check imports
            broken = check_imports(f)
            if broken:
                print(f"\n[BROKEN IMPORTS] {rel_path}")
                for imp in broken:
                    print(f"  - {imp}")

            # Check file refs
            missing = check_file_references(f, project_root)
            if missing:
                print(f"\n[MISSING FILES] {rel_path}")
                for m in missing:
                    print(f"  - {m}")


if __name__ == "__main__":
    main()
