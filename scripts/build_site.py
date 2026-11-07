"""Build index.html from proposal.md with Python's standard library."""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', r'<a href="\2">\1</a>', text)
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


def render(text):
    output = []
    paragraph = []
    current_list = None
    table = False

    def flush():
        if paragraph:
            output.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    for line in text.splitlines() + ['']:
        line = line.strip()
        item = re.match(r'^(?:(\d+)\.|(-))\s+(.+)', line)
        kind = ('ol' if item[1] else 'ul') if item else None
        if current_list and kind != current_list:
            output.append(f'</{current_list}>')
            current_list = None
        if table and not line.startswith('|'):
            output.append('</tbody></table></div>')
            table = False
        if line.startswith('|'):
            flush()
            cells = [c.strip() for c in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?', c) for c in cells):
                continue
            tag = 'td' if table else 'th'
            if not table:
                output.append('<div class="table-wrap"><table><thead>')
            output.append('<tr>' + ''.join(f'<{tag}>' + inline(c) + f'</{tag}>' for c in cells) + '</tr>')
            if not table:
                output.append('</thead><tbody>')
                table = True
        elif item:
            flush()
            if not current_list:
                output.append(f'<{kind}>')
                current_list = kind
            output.append('<li>' + inline(item[3]) + '</li>')
        elif line:
            paragraph.append(line)
        else:
            flush()
    return '\n'.join(output)


source = (ROOT / 'proposal.md').read_text()
title = source.splitlines()[0].removeprefix('# ')
parts = re.split(r'^## (.+)\s*$', source, flags=re.MULTILINE)
navigation, sections = [], []
for i in range(1, len(parts), 2):
    heading = parts[i].strip().rstrip(':').strip()
    anchor = re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')
    navigation.append(f'<a href="#{anchor}">{html.escape(heading)}</a>')
    sections.append(f'<section id="{anchor}"><span class="number">{(i+1)//2:02}</span><div><h2>{html.escape(heading)}</h2>{render(parts[i+1])}</div></section>')

page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A 15-618 project proposal exploring CUDA optimizations for multi-scale exposure fusion, by Estella Xie and Junhao Li.">
<title>''' + html.escape(title) + '''</title><link rel="stylesheet" href="assets/style.css"></head>
<body><a class="skip" href="#proposal">Skip to proposal</a>
<main><div class="hero">
<h1>''' + html.escape(title) + '''</h1>
<div class="team"><div><span>TEAM</span>Estella Xie <small>ziyoux</small><br>Junhao Li <small>junhaoli</small></div></div></div>
<div class="document" id="proposal"><aside><div class="eyebrow">IN THIS PROPOSAL</div><nav aria-label="Proposal sections">''' + ''.join(navigation) + '''</nav><a href="proposal.md">Markdown source ↗</a></aside><article>''' + '\n'.join(sections) + '''</article></div></main>
<footer>Estella Xie &amp; Junhao Li · 15-618 · CUDA-Accelerated Multi-Exposure Image Fusion</footer></body></html>
'''
(ROOT / 'index.html').write_text(page)
print('Built index.html from proposal.md')
