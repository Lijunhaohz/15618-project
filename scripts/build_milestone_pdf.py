#!/usr/bin/env python3
"""Build docs/milestone.pdf from the report Markdown and curated measured figures."""
import argparse
import csv
from collections import defaultdict
from html import escape
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from plot_results import load, median, CPU, CUDA, STAGES
import matplotlib.pyplot as plt


def figures(results):
    cases, memory, _ = load(results)
    data = sorted((c for c in cases.values() if c['level'] == 5 and c['n'] in [3, 5, 9]), key=lambda c: c['n'])
    if len(data) != 3:
        raise ValueError('Milestone tables expect the saved Memorial N=3,5,9 / L=5 run')
    target = ROOT / 'docs/imgs'
    target.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout='constrained')
    x = list(range(len(data)))
    for offset, metric, color, label in [(-.18, CPU, '#596675', 'CPU'), (.18, CUDA, '#217b9d', 'CUDA')]:
        axes[0].bar([v + offset for v in x], [median(c, metric) for c in data], width=.35, color=color, label=label)
    axes[0].set(xticks=x, xticklabels=[str(c['n']) for c in data], xlabel='Exposure count', ylabel='Median wall time (ms)', title='Fusion call, five levels')
    axes[0].legend()
    speedup = [median(c, CPU) / median(c, CUDA) for c in data]
    axes[1].bar(x, speedup, color='#217b9d')
    for i, value in enumerate(speedup):
        axes[1].text(i, value + .15, f'{value:.2f}×', ha='center')
    axes[1].set(xticks=x, xticklabels=[str(c['n']) for c in data], xlabel='Exposure count', ylabel='CPU / CUDA wall time', ylim=(0, 11), title='Matched wall-time speedup')
    fig.savefig(target / 'milestone-runtime.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout='constrained')
    focus = data[0]
    axes[0].barh([label for _, label in reversed(STAGES)], [median(focus, metric) for metric, _ in reversed(STAGES)], color='#217b9d')
    axes[0].set(xlabel='Median event interval (ms)', title='Three exposures, five levels')
    axes[1].plot([c['n'] for c in data], [max(int(r['peak_bytes']) for r in memory[c['name']]) / 2**20 for c in data], 'o-', color='#217b9d')
    axes[1].set(xlabel='Exposure count', ylabel='Peak requested GPU buffers (MiB)', title='Resident baseline, five levels', xticks=[3, 5, 9], ylim=(0, 160))
    fig.savefig(target / 'milestone-stages-memory.png', dpi=180)
    plt.close(fig)


def pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    for name, filename in [('Report', 'DejaVuSans.ttf'), ('Report-Bold', 'DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily('Report', normal='Report', bold='Report-Bold', italic='Report', boldItalic='Report-Bold')
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'Report'
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    styles['BodyText'].spaceAfter = 6
    styles['Title'].fontSize = 19
    styles['Title'].leading = 24
    styles['Heading2'].fontName = 'Report-Bold'
    styles['Heading2'].fontSize = 12
    styles['Heading2'].leading = 16
    styles['Heading2'].spaceBefore = 12
    styles['Heading3'].fontName = 'Report-Bold'
    styles['Heading3'].fontSize = 10
    styles['Heading3'].leading = 14
    styles.add(ParagraphStyle('Cell', parent=styles['BodyText'], fontSize=7.5, leading=10, spaceAfter=0))
    styles.add(ParagraphStyle('CaptionSmall', parent=styles['BodyText'], fontSize=8, leading=11, alignment=TA_CENTER))
    target = ROOT / 'docs/milestone.pdf'
    doc = SimpleDocTemplate(str(target), pagesize=(612, 792), rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40, title='CUDA-Accelerated Multi-Exposure Image Fusion — Milestone',
                            author='Estella Xie and Junhao Li')
    width = 532

    def inline(text):
        text = escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: '<link href="' + escape(m[2], quote=True) + '" color="#217b9d">' + m[1] + '</link>', text)

    source = (ROOT / 'docs/milestone.md').read_text().splitlines()
    story, paragraph = [], []

    def flush():
        if paragraph:
            story.append(Paragraph(inline(' '.join(paragraph)), styles['BodyText']))
            paragraph.clear()
    i = 0
    while i < len(source):
        line = source[i].strip()
        i += 1
        if not line:
            flush()
        elif line.startswith('#'):
            flush()
            match = re.match(r'^(#+)\s+(.*)$', line)
            style = 'Title' if len(match[1]) == 1 else 'Heading' + str(min(len(match[1]), 3))
            story.append(Paragraph(inline(match[2]), styles[style]))
        elif line.startswith('|'):
            flush()
            rows = [line]
            while i < len(source) and source[i].strip().startswith('|'):
                rows.append(source[i].strip())
                i += 1
            cells = [[c.strip() for c in row.strip('|').split('|')] for row in rows]
            cells = [row for row in cells if not all(re.fullmatch(r':?-+:?', c) for c in row)]
            count = len(cells[0])
            if count == 3 and cells[0][0] == 'Dates':
                widths = [82, 225, 225]
            else:
                widths = [width / count] * count
            values = [[Paragraph(('<b>' + inline(c) + '</b>') if n == 0 else inline(c), styles['Cell']) for c in row] for n, row in enumerate(cells)]
            table = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eff3')),
                ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#607987')),
                ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#dce3e7')),
                ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
            story.extend([Spacer(1, 5), table, Spacer(1, 7)])
        elif line.startswith('!['):
            flush()
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
            path = ROOT / 'docs' / match[2]
            iw, ih = ImageReader(str(path)).getSize()
            story.append(KeepTogether([Image(str(path), width=width, height=width * ih / iw),
                                      Paragraph(inline(match[1]), styles['CaptionSmall']), Spacer(1, 7)]))
        elif re.match(r'^\d+\.\s', line):
            flush()
            story.append(Paragraph(inline(line), styles['BodyText']))
        else:
            paragraph.append(line)
    flush()

    def footer(canvas, document):
        canvas.setFont('Report', 7)
        canvas.setFillColor(colors.HexColor('#55616a'))
        canvas.drawRightString(572, 22, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Built ' + str(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, help='regenerate curated figures from the measured run')
    args = parser.parse_args()
    if args.results:
        figures(args.results)
    pdf()
