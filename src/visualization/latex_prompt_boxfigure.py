from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert prompt JSON to LaTeX figure.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input prompt template JSON file",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output LaTeX file",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Prompt Template.",
        help="Caption for the figure",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="fig:prompt_template",
        help="Label for the figure",
    )
    return parser.parse_args()


def escape_latex_text(text: str) -> str:
    """Escapes core LaTeX special characters."""
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}'
    }
    res = []
    for c in text:
         res.append(chars.get(c, c))
    return "".join(res)

def parse_markdown_to_latex(content: str) -> str:
    """Converts markdown features to LaTeX."""
    # Split by code blocks
    parts = re.split(r'(```.*?```)', content, flags=re.DOTALL)
    latex_parts = []
    
    for part in parts:
        if part.startswith('```'):
            match = re.match(r'```(\w*)\n(.*?)```', part, flags=re.DOTALL)
            if match:
                code = match.group(2).strip()
            else:
                code = part[3:-3].strip()
            latex_parts.append(f"\n\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}\n")
        else:
            text = escape_latex_text(part)
            lines = text.split('\n')
            processed_lines = []
            for line in lines:
                # Handle markdown headers like #### Header ####
                header_match = re.match(r'^\\#+\s+(.*?)(?:\s+\\#+)?$', line.strip())
                if header_match:
                    header_text = header_match.group(1)
                    processed_lines.append(f"\\par\\vspace{{2mm}}\\noindent\\textbf{{{header_text}}}\\par")
                elif line.strip() == '':
                    processed_lines.append('\\par')
                else:
                    # Inline code
                    processed_line = re.sub(r'`([^`]+)`', r'\\texttt{\1}', line)
                    processed_lines.append(processed_line)
            latex_parts.append('\n'.join(processed_lines))
            
    return "".join(latex_parts)

def convert_to_latex(prompt_data: list[dict[str, Any]], caption: str = "Prompt template", label: str = "fig:prompt_template") -> str:
    """Creates a LaTeX figure with a tcolorbox from prompt data."""
    system_content = ""
    user_content = ""
    
    for msg in prompt_data:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_content += parse_markdown_to_latex(content) + "\n"
        elif role == "user":
            user_content += parse_markdown_to_latex(content) + "\n"
            
    latex_code = [
        "\\begin{figure}[htbp]",
        "\\centering",
        "\\begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=Prompt Template,arc=2mm,boxrule=0.5mm]",
    ]
    
    if system_content:
        latex_code.append("\\noindent\\textbf{\\Large System}\\par")
        latex_code.append("\\vspace{2mm}")
        latex_code.append(system_content.strip())
        latex_code.append("\\vspace{4mm}")
        
    if user_content:
        if system_content:
            latex_code.append("\\tcblower")
        latex_code.append("\\noindent\\textbf{\\Large User}\\par")
        latex_code.append("\\vspace{2mm}")
        latex_code.append(user_content.strip())
        
    latex_code.append("\\end{tcolorbox}")
    latex_code.append(f"\\caption{{{caption}}}")
    latex_code.append(f"\\label{{{label}}}")
    latex_code.append("\\end{figure}")
    
    return "\n".join(latex_code)

def main() -> None:
    """Run prompt-to-LaTeX conversion."""
    args = parse_args()
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
        
    latex_str = convert_to_latex(prompt_data, caption=args.caption, label=args.label)
    
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
        
    print(f"Successfully wrote LaTeX snippet to {args.output_path}")

if __name__ == "__main__":
    main()
