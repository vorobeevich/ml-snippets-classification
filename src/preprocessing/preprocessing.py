import re
from typing import List, Union


def from_bytes_to_str(snippet: str) -> str:
    """Converts a string from byte to regular format."""
    format_ = f'b"'
    format_hat = f"b'"
    if snippet[:2] == format_ or snippet[:2] == format_hat:
        n = len(snippet)
        snippet = snippet[2: n - 1]
    return snippet


def correct_special_symbols(snippet: str) -> str:
    """Fixes special characters."""
    snippet = snippet.replace('\\n', '\n')
    snippet = snippet.replace('<br>', '\n')
    snippet = snippet.replace('\n', ' \n ')
    snippet = snippet.replace('\\"', '\"')
    snippet = snippet.replace("\\'", "\'")
    snippet = snippet.replace('\'\'\'', ' \'\'\' ')
    snippet = snippet.replace('\"\"\"', ' \"\"\" ')
    return snippet


def delete_short_comments(snippet: str) -> str:
    """Deletes short comments."""
    snippet = re.sub('#.*', '', snippet)
    return snippet


def delete_imports(snippet: str) -> str:
    """Deletes imports modules from code."""
    im_snippet = re.sub('from .* import .*', '', snippet)
    im_snippet = re.sub('import .*', '', im_snippet)
    if len(im_snippet.replace(' ', '').replace('\n', '')) != 0:
        snippet = im_snippet
    return snippet


def add_spaces(snippet: str, symbols: List[str]) -> str:
    """Adds spaces to the beginning and end of each character."""
    for symb in symbols:
        snippet = snippet.replace(symb, ' ' + symb + ' ')
    return snippet


def delete_empty_lines(snippet: str) -> List[str]:
    """Deletes empty lines in snippet."""
    snippet = snippet.split(sep=' ')
    while '' in snippet:
        snippet.remove('')

    if len(snippet) == 0:
        snippet = ['\n']

    new_snippet = [snippet[0]]
    for i in range(1, len(snippet)):
        if snippet[i] == '\n' and new_snippet[-1] == '\n':
            continue
        else:
            new_snippet.append(snippet[i])
    if len(new_snippet) > 1 and new_snippet[0] == '\n':
        new_snippet = new_snippet[1:]
    return new_snippet


def delete_long_comments(snippet: List[str]) -> List[str]:
    """Deletes long comments."""
    res_snippet = []
    comment_flag = 0
    opened = ''
    for item in snippet:
        if item == '\'\'\'' or item == "\\'\\'\\'" or item == '\"\"\"' or item == '\\"\\"\\"':
            comment_flag += 1
            comment_flag %= 2
            if comment_flag == 1:
                opened = item
            else:
                if item != opened:
                    comment_flag = 1
            continue
        if comment_flag != 1:
            res_snippet.append(item)
    if len(res_snippet) == 0:
        res_snippet = ['\n']
    return res_snippet


def preprocess_snippet(snippet: str, format='list') -> Union[str, List[str]]:
    """Performs string preprocessing."""
    functions = [from_bytes_to_str, correct_special_symbols,
                 delete_short_comments, delete_imports]
    for function in functions:
        snippet = function(snippet)
    to_replace = ['.', '(', ')', '\n', '[', ']', '_']
    snippet = add_spaces(snippet, to_replace)
    new_snippet = delete_empty_lines(snippet)
    res_snippet = delete_long_comments(new_snippet)
    if format == 'str':
        return ' '.join(res_snippet)
    return res_snippet
