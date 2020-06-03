#!/usr/bin/env python3
from collections import defaultdict
import os
import re

import click

COLUMN_ORDER = ('Demo', 'TestJitter', 'TestLayout', 'TestColour', 'TestShape',
                'TestCountPlus', 'TestDynamics', 'TestAll')
PREAMBLE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Run videos</title>
  </head>
  <body>
"""
ENDING = """
  </body>
</html>
"""
DATE_SUFFIX_RE = re.compile(r'-\d{4}-\d{2}-\d{2}$')
VID_RE = re.compile('^vid-(?P<task>.+)-(?P<variant>.+).mp4$')


def build_index(root):
    """Build a three-level index of nested dicts. Mapping is task -> algorithm
    -> variant -> video path (where video paths are the leaves of the
    structure). e.g. use `index[task][alg][var]` to get a video path for
    algorithm `alg` on task `task` and variant `var`."""
    index = defaultdict(lambda: defaultdict(dict))

    alg_bnames = os.listdir(root)

    for alg_bname in alg_bnames:
        # algorithm name just removes date
        alg = DATE_SUFFIX_RE.sub('', alg_bname)
        alg_dir = os.path.join(root, alg_bname)
        if not os.path.isdir(alg_dir):
            continue
        for video_bname in os.listdir(alg_dir):
            # this path is relative to the root
            video_path = os.path.join(alg_bname, video_bname)
            match = VID_RE.match(video_bname)
            task = match.group('task')
            variant = match.group('variant')
            index[task][alg][variant] = video_path

    return index


def render_table(task, task_dict, fp):
    # print lead row
    print("<table>", file=fp)
    print("<tr>", file=fp)
    for col in ('Algorithm', ) + COLUMN_ORDER:
        print(f"<th>{col}</th>", file=fp)
    print("</tr>", file=fp)

    for alg, alg_cols in sorted(task_dict.items()):
        print("<tr>", file=fp)

        for column in COLUMN_ORDER:
            vid_path = alg_cols.get(column.lower())
            print("<td>", file=fp)
            if not vid_path:
                contents = "-"
            else:
                contents = "<video playsinline muted autoplay loop " \
                    "controls width='320' height='128'>" \
                    f"<source src='{vid_path}' type='video/mp4' />" \
                    "</video>"
            print(contents, file=fp)
            print("</td>", file=fp)

        print("</tr>", file=fp)

    print("</table>", file=fp)


@click.command()
@click.argument('root')
def main(root):
    """The `render_model_vids.py` script will create a huge directory of videos
    for every available model & variant. This collects those videos into a
    series of HTML tables that you can view in your browser."""
    print("Building index")
    index = build_index(root)

    with open(os.path.join(root, "index.html"), "w") as fp:
        fp.write(PREAMBLE)
        for task, task_dict in sorted(index.items()):
            print("Processing task", task)
            print(f"<h1>Results on {task}</h1>", file=fp)
            render_table(task, task_dict, fp)
        fp.write(ENDING)

    print(f"Done, try `pushd {root} && python3 -m http.server; popd`")


if __name__ == '__main__':
    main()
