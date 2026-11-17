"""Build the milestone project page and archived proposal using the standard library."""
import html
import re
import posixpath
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = html.escape(text)
    def link(match):
        url = match[2]
        if not url.startswith(('https://', 'http://', '#')):
            if ':' in url:
                return match[1]
            url = posixpath.normpath('docs/' + url)
        return '<a href="' + url + '">' + match[1] + '</a>'
    text = re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', link, text)
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

    lines = text.splitlines() + ['']
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        quote = re.match(r'^>\s?(.*)$', line)
        if quote:
            flush()
            if current_list:
                output.append(f'</{current_list}>')
                current_list = None
            if table:
                output.append('</tbody></table></div>')
                table = False
            quoted_lines = [quote[1]]
            while i < len(lines):
                quote = re.match(r'^>\s?(.*)$', lines[i].strip())
                if not quote:
                    break
                quoted_lines.append(quote[1])
                i += 1
            output.append('<blockquote>' + render('\n'.join(quoted_lines)) + '</blockquote>')
            continue

        heading = re.match(r'^(#{3,6})\s+(.+?)\s*#*\s*$', line)
        rule = re.fullmatch(r'(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,}', line)
        item = re.match(r'^(?:(\d+)\.|(-))\s+(.+)', line)
        kind = ('ol' if item[1] else 'ul') if item and not rule else None
        if current_list and kind != current_list:
            output.append(f'</{current_list}>')
            current_list = None
        if table and not line.startswith('|'):
            output.append('</tbody></table></div>')
            table = False
        figure = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if figure:
            flush()
            path = posixpath.normpath('docs/' + figure[2])
            output.append('<figure><img style="width:100%;height:auto" src="' + html.escape(path, quote=True) + '" alt="' + html.escape(figure[1], quote=True) + '"><figcaption>' + inline(figure[1]) + '</figcaption></figure>')
        elif heading:
            flush()
            level = len(heading[1])
            output.append(f'<h{level}>' + inline(heading[2]) + f'</h{level}>')
        elif rule:
            flush()
            output.append('<hr>')
        elif line.startswith('|'):
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
        elif kind:
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


def build(source_path, output_path, label):
    source = (ROOT / source_path).read_text()
    title = source.splitlines()[0].removeprefix('# ')
    parts = re.split(r'^## (.+)\s*$', source, flags=re.MULTILINE)
    intro = '\n'.join(parts[0].splitlines()[1:])
    navigation, sections = [], []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().rstrip(':').strip()
        anchor = re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')
        navigation.append(f'<a href="#{anchor}">{html.escape(heading)}</a>')
        sections.append(f'<section id="{anchor}"><span class="number">{(i+1)//2:02}</span><div><h2>{html.escape(heading)}</h2>{render(parts[i+1])}</div></section>')
    is_proposal = source_path == 'docs/proposal.md'
    document_id = 'proposal' if is_proposal else 'milestone'
    tabs = []
    for href, name, active in [('index.html', 'Proposal', is_proposal),
                               ('milestone.html', 'Milestone Report', not is_proposal)]:
        text = '<strong>' + name + '</strong>' if active else name
        current = ' aria-current="page"' if active else ''
        tabs.append('<a href="' + href + '"' + current + '>' + text + '</a>')
    tabs_html = '<header aria-label="Project documents"><div>' + ' &nbsp; / &nbsp; '.join(tabs) + '</div></header>'
    project_url = re.search(r'(?m)^\*\*Project URL\*\*[ \t]*\n(?:[ \t]*\n)*[^\n]+', intro)
    url_html = render(project_url[0]) if project_url else '<p><strong>Project URL</strong></p><p><a href="https://Lijunhaohz.github.io/15618-project/">https://Lijunhaohz.github.io/15618-project/</a></p>'
    team_html = '<div class="team"><div><span>TEAM</span>Estella Xie <small>ziyoux</small><br>Junhao Li <small>junhaoli</small></div><div class="project-url">' + url_html + '</div></div>'
    report_intro = '' if is_proposal else render('\n'.join(line for line in intro.splitlines() if not line.startswith('**Estella Xie')))
    pdf_path = 'docs/proposal.pdf' if is_proposal else 'docs/milestone.pdf'
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page += '<meta name="description" content="15-618 exposure fusion project: implementation progress, measured results, and revised schedule.">'
    page += '<title>' + html.escape(title + ' — ' + label) + '</title><link rel="stylesheet" href="assets/style.css"></head>'
    page += '<body><a class="skip" href="#' + document_id + '">Skip to ' + label.lower() + '</a>' + tabs_html
    page += '<main><div class="hero"><h1>' + html.escape(title) + '</h1>' + team_html + report_intro
    page += '</div><div class="document" id="' + document_id + '"><aside><div class="eyebrow">' + html.escape('IN THIS ' + label.upper()) + '</div><nav aria-label="' + html.escape(label, quote=True) + ' sections">'
    page += ''.join(navigation) + '</nav><a href="' + pdf_path + '">Download PDF ↗</a><br><a href="' + source_path + '">Markdown source ↗</a></aside><article>'
    page += '\n'.join(sections) + '</article></div></main><footer>Estella Xie &amp; Junhao Li · 15-618 · CUDA-Accelerated Multi-Exposure Image Fusion</footer></body></html>'
    (ROOT / output_path).write_text(page)
    print('Built ' + output_path + ' from ' + source_path)


if __name__ == '__main__':
    build('docs/proposal.md', 'index.html', 'Proposal')
    build('docs/proposal.md', 'proposal.html', 'Proposal')
    build('docs/milestone.md', 'milestone.html', 'Milestone Report')
