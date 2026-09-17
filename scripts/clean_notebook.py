import sys
import json
import os

def clean_notebook(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False

    print(f"Processing notebook: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        return False

    changed = False
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'markdown':
            lines = cell.get('source', [])
            new_lines = []
            cell_changed = False
            for line in lines:
                original = line
                # 1. Replace \tag{X} or \tag*{(X)}
                for num in range(1, 30):
                    line = line.replace(f'\\tag{{{num}}}', '')
                    line = line.replace(f'\\tag*{{({num})}}', '')
                
                # 2. Replace \newline inside math
                line = line.replace('\\newline\\;', '\\\\')
                line = line.replace('\\newline', '\\\\')

                if line != original:
                    cell_changed = True
                new_lines.append(line)

            if cell_changed:
                cell['source'] = new_lines
                changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully cleaned KaTeX syntax in '{file_path}'!")
    else:
        print(f"No KaTeX parse issues found in '{file_path}'.")

    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean_notebook.py <path_to_notebook.ipynb>")
        sys.exit(1)

    clean_notebook(sys.argv[1])
