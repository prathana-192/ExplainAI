import os
import io
import nbformat

MAX_CODE_FILE_SIZE_MB = 5 

def parse_uploaded_code_file(uploaded_file) -> tuple[str, str]:
    if uploaded_file is None:
        return "", "No file uploaded."

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_CODE_FILE_SIZE_MB:
        return "", f"File size exceeds {MAX_CODE_FILE_SIZE_MB}MB limit."

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    code_content = ""
    error_message = ""

    if file_extension == '.py':
        try:
            code_content = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            error_message = f"Failed to read Python file: {e}."
    elif file_extension == '.ipynb':
        try:
            notebook_content = uploaded_file.getvalue().decode("utf-8")
            notebook = nbformat.read(io.StringIO(notebook_content), as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    code_content += cell.source + "\n\n"
            if not code_content.strip():
                error_message = "No executable code cells found."
        except Exception as e:
            error_message = f"Failed to parse Jupyter notebook: {e}."
    else:
        error_message = f"Unsupported file type: {file_extension}."

    return code_content, error_message
