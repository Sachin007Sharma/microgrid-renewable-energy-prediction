#!/usr/bin/env python3
"""
convert_report_to_docx.py
Converts Markdown report to a beautifully styled Microsoft Word (.docx) document.
Built using pure Python standard library (zipfile, xml.sax.saxutils).
Outputs: report/final_report.docx
"""

import os
import re
import io
import zipfile
from xml.sax.saxutils import escape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "report", "final_report.md")
DOCX_PATH = os.path.join(BASE_DIR, "report", "final_report.docx")

# ── XML Building Blocks ───────────────────────────────────────────────────────

CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
</Types>'''

RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

DOC_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>
</Relationships>'''

SETTINGS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:defaultTabStop w:val="720"/>
  <w:compat>
    <w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/>
  </w:compat>
</w:settings>'''

FONT_TABLE = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fontTable xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:font w:name="Times New Roman">
    <w:panose1 w:val="02020603050405020304"/>
    <w:charset w:val="00"/>
    <w:family w:val="roman"/>
    <w:pitch w:val="variable"/>
  </w:font>
  <w:font w:name="Calibri">
    <w:panose1 w:val="020F0502020204030204"/>
    <w:charset w:val="00"/>
    <w:family w:val="swiss"/>
    <w:pitch w:val="variable"/>
  </w:font>
  <w:font w:name="Consolas">
    <w:panose1 w:val="020B0609020204030204"/>
    <w:charset w:val="00"/>
    <w:family w:val="modern"/>
    <w:pitch w:val="fixed"/>
  </w:font>
</w:fontTable>'''

STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:color w:val="1A202C"/>
        <w:lang w:val="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="276" w:lineRule="auto" w:after="120"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>

  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="240" w:after="180"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="48"/>
      <w:color w:val="1B365D"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="60" w:after="240"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:i/>
      <w:sz w:val="24"/>
      <w:color w:val="4A5568"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="360" w:after="140"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="32"/>
      <w:color w:val="1B365D"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="280" w:after="100"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="26"/>
      <w:color w:val="2B6CB0"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="200" w:after="80"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:i/>
      <w:sz w:val="23"/>
      <w:color w:val="2D3748"/>
    </w:rPr>
  </w:style>
</w:styles>'''


def format_runs(text):
    """
    Parses inline markdown like **bold**, *italic*, `code`, and plain text.
    Returns OpenXML <w:r> elements.
    """
    runs_xml = []
    # Tokenize by bold, italic, code, or math
    pattern = re.compile(r'(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+?`|\$[^\$]+?\$|[^\*`\$]+)')
    tokens = pattern.findall(text)
    if not tokens:
        tokens = [text]

    for tok in tokens:
        if not tok:
            continue
        is_bold = False
        is_italic = False
        is_code = False
        inner = tok

        if tok.startswith("**") and tok.endswith("**") and len(tok) >= 4:
            is_bold = True
            inner = tok[2:-2]
        elif tok.startswith("*") and tok.endswith("*") and len(tok) >= 2:
            is_italic = True
            inner = tok[1:-1]
        elif tok.startswith("`") and tok.endswith("`") and len(tok) >= 2:
            is_code = True
            inner = tok[1:-1]
        elif tok.startswith("$") and tok.endswith("$") and len(tok) >= 2:
            # Inline math: format in clean italic Serif font
            is_italic = True
            inner = tok[1:-1]

        rpr_parts = []
        if is_code:
            rpr_parts.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
            rpr_parts.append('<w:sz w:val="19"/>')
            rpr_parts.append('<w:color w:val="9B2C2C"/>')
            rpr_parts.append('<w:shd w:val="clear" w:color="auto" w:fill="EDF2F7"/>')
        else:
            if is_bold:
                rpr_parts.append('<w:b/>')
            if is_italic:
                rpr_parts.append('<w:i/>')

        rpr_xml = f'<w:rPr>{"".join(rpr_parts)}</w:rPr>' if rpr_parts else ''
        escaped_txt = escape(inner)
        runs_xml.append(f'<w:r>{rpr_xml}<w:t xml:space="preserve">{escaped_txt}</w:t></w:r>')

    return "".join(runs_xml)


def make_p(text, style=None, align=None, is_callout=False, is_bullet=False):
    """Generates a <w:p> XML element."""
    ppr_parts = []
    if style:
        ppr_parts.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr_parts.append(f'<w:jc w:val="{align}"/>')

    if is_callout:
        ppr_parts.append('<w:pBdr><w:left w:val="single" w:sz="24" w:space="15" w:color="3182CE"/></w:pBdr>')
        ppr_parts.append('<w:shd w:val="clear" w:color="auto" w:fill="F7FAFC"/>')
        ppr_parts.append('<w:ind w:left="360" w:right="360"/>')

    if is_bullet:
        ppr_parts.append('<w:ind w:left="720" w:hanging="360"/>')
        bullet_run = '<w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:color w:val="3182CE"/></w:rPr><w:t>▪ </w:t></w:r>'
    else:
        bullet_run = ''

    ppr_xml = f'<w:pPr>{"".join(ppr_parts)}</w:pPr>' if ppr_parts else ''
    body_runs = format_runs(text)
    return f'<w:p>{ppr_xml}{bullet_run}{body_runs}</w:p>'


def make_table(rows):
    """
    Renders a Markdown table into an OpenXML table with Navy header and zebra borders.
    """
    if not rows:
        return ""

    num_cols = max(len(r) for r in rows)
    # 9360 twips = 6.5 inches printable width
    col_w = int(9360 / num_cols)

    tbl_xml = ['<w:tbl>']
    tbl_xml.append('''<w:tblPr>
      <w:tblW w:w="9360" w:type="dxa"/>
      <w:jc w:val="center"/>
      <w:tblBorders>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
        <w:left w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
        <w:bottom w:val="single" w:sz="8" w:space="0" w:color="CBD5E1"/>
        <w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="120" w:type="dxa"/>
        <w:left w:w="160" w:type="dxa"/>
        <w:bottom w:w="120" w:type="dxa"/>
        <w:right w:w="160" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>''')

    # Grid
    tbl_xml.append('<w:tblGrid>')
    for _ in range(num_cols):
        tbl_xml.append(f'<w:gridCol w:w="{col_w}"/>')
    tbl_xml.append('</w:tblGrid>')

    for row_idx, r in enumerate(rows):
        is_header = (row_idx == 0)
        bg_color = "1B365D" if is_header else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        txt_color = "FFFFFF" if is_header else "1A202C"

        tbl_xml.append('<w:tr>')
        tbl_xml.append(f'<w:trPr><w:cantSplit/>{"<w:tblHeader/>" if is_header else ""}</w:trPr>')

        for cell_idx in range(num_cols):
            cell_text = r[cell_idx] if cell_idx < len(r) else ""
            cell_text = cell_text.strip()

            tc_parts = []
            tc_parts.append(f'<w:tcW w:w="{col_w}" w:type="dxa"/>')
            tc_parts.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{bg_color}"/>')

            # cell paragraph
            p_align = "center" if (is_header or cell_idx > 1) else "left"
            if is_header:
                formatted_p = f'<w:p><w:pPr><w:jc w:val="{p_align}"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:sz w:val="20"/><w:color w:val="{txt_color}"/></w:rPr><w:t>{escape(cell_text)}</w:t></w:r></w:p>'
            else:
                formatted_runs = format_runs(cell_text)
                formatted_p = f'<w:p><w:pPr><w:jc w:val="{p_align}"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>{formatted_runs}</w:p>'

            tbl_xml.append(f'<w:tc><w:tcPr>{"".join(tc_parts)}</w:tcPr>{formatted_p}</w:tc>')

        tbl_xml.append('</w:tr>')

    tbl_xml.append('</w:tbl>')
    # Spacing after table
    tbl_xml.append('<w:p><w:pPr><w:spacing w:after="140"/></w:pPr></w:p>')
    return "".join(tbl_xml)


def make_code_block(code_lines):
    """Renders formatted monospace box for ASCII art or code listings."""
    xml_parts = []
    xml_parts.append('<w:p><w:pPr>')
    xml_parts.append('<w:pBdr><w:top w:val="single" w:sz="4" w:space="4" w:color="CBD5E1"/><w:left w:val="single" w:sz="18" w:space="8" w:color="2B6CB0"/><w:bottom w:val="single" w:sz="4" w:space="4" w:color="CBD5E1"/><w:right w:val="single" w:sz="4" w:space="4" w:color="CBD5E1"/></w:pBdr>')
    xml_parts.append('<w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/>')
    xml_parts.append('<w:ind w:left="200" w:right="200"/>')
    xml_parts.append('<w:spacing w:before="120" w:after="120" w:line="220" w:lineRule="auto"/>')
    xml_parts.append('</w:pPr>')

    for idx, line in enumerate(code_lines):
        escaped_line = escape(line)
        xml_parts.append(f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/><w:color w:val="2D3748"/></w:rPr><w:t xml:space="preserve">{escaped_line}</w:t></w:r>')
        if idx < len(code_lines) - 1:
            xml_parts.append('<w:r><w:br/></w:r>')

    xml_parts.append('</w:p>')
    return "".join(xml_parts)


# ── Main Markdown Parser ──────────────────────────────────────────────────────

def parse_markdown_to_docx(md_path, docx_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    body_xml = []
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")

        # Code block fence
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                body_xml.append(make_code_block(code_lines))
                code_lines = []
            else:
                if in_table:
                    body_xml.append(make_table(table_rows))
                    table_rows = []
                    in_table = False
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Table detection
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            # Check if separator row
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            in_table = True
            i += 1
            continue
        else:
            if in_table:
                body_xml.append(make_table(table_rows))
                table_rows = []
                in_table = False

        stripped = line.strip()

        # Empty line
        if not stripped:
            i += 1
            continue

        # Horizontal Rule
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            body_xml.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="8" w:color="E2E8F0"/></w:pBdr><w:spacing w:before="120" w:after="160"/></w:pPr></w:p>')
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            body_xml.append(make_p(stripped[2:], style="Title"))
        elif stripped.startswith("## "):
            body_xml.append(make_p(stripped[3:], style="Heading1"))
        elif stripped.startswith("### "):
            body_xml.append(make_p(stripped[4:], style="Heading2"))
        elif stripped.startswith("#### "):
            body_xml.append(make_p(stripped[5:], style="Heading3"))
        elif stripped.startswith("> "):
            body_xml.append(make_p(stripped[2:], is_callout=True))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            body_xml.append(make_p(stripped[2:], is_bullet=True))
        elif re.match(r'^\d+\.\s+', stripped):
            num_match = re.match(r'^(\d+\.\s+)(.*)$', stripped)
            prefix = num_match.group(1)
            content = num_match.group(2)
            body_xml.append(make_p(f"**{prefix}**{content}"))
        else:
            body_xml.append(make_p(stripped))

        i += 1

    # End any pending table or code
    if in_table:
        body_xml.append(make_table(table_rows))
    if in_code_block:
        body_xml.append(make_code_block(code_lines))

    # Section properties: Standard Letter / A4 with 1-inch margins (1440 twips)
    body_xml.append('''<w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
      <w:cols w:space="720"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>''')

    full_document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {"".join(body_xml)}
  </w:body>
</w:document>'''

    # Assemble ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("word/_rels/document.xml.rels", DOC_RELS)
        zf.writestr("word/settings.xml", SETTINGS)
        zf.writestr("word/fontTable.xml", FONT_TABLE)
        zf.writestr("word/styles.xml", STYLES)
        zf.writestr("word/document.xml", full_document_xml)

    with open(docx_path, "wb") as f:
        f.write(buf.getvalue())

    print(f"✅ Success! Created {docx_path} ({len(buf.getvalue()):,} bytes)")


if __name__ == "__main__":
    parse_markdown_to_docx(MD_PATH, DOCX_PATH)

