#!/usr/bin/env python3
from collections import OrderedDict, defaultdict
import os
import re

import click

COLUMN_ORDER = ('Demo', 'TestJitter', 'TestLayout', 'TestColour', 'TestShape',
                'TestCountPlus', 'TestDynamics', 'TestAll')
PREAMBLE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>%s</title>
    <script>
        function syncVids() {
            let videos = document.querySelectorAll("tr:not(.hidden) video");
            for (let vid of videos) {
                vid.currentTime = 0; vid.play();
            }
        }

        function toggleShow(anchor) {
            let elem = anchor;
            while (elem.tagName != "TR" && elem != null) {
                elem = elem.parentElement;
            }
            if (elem == null) {
                console.error("could not find parent node of " + anchor);
            }
            let subVids = elem.querySelectorAll("video");
            if (elem.classList.contains("hidden")) {
                elem.classList.remove("hidden");
                for (let v of subVids) {
                    v.currentTime = 0;
                    v.play();
                }
            } else {
                elem.classList.add("hidden");
                for (let v of subVids) {
                    v.pause();
                }
            }
        }
    </script>

    <style>
        .hidden video {
            display: none;
        }
        .hidden td {
            background-colour: #aaa;
        }
    </style>
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

        print(f'<th><a href="#" onclick="toggleShow(this)">{alg}</a></th>',
              file=fp)

        for column in COLUMN_ORDER:
            vid_path = alg_cols.get(column.lower())
            print("<td>", file=fp)
            if not vid_path:
                contents = "-"
            else:
                contents = "<video playsinline muted loop " \
                    "controls width='320' height='128'>" \
                    f"<source src='{vid_path}' type='video/mp4' />" \
                    "</video>"
            print(contents, file=fp)
            print("</td>", file=fp)

        print("</tr>", file=fp)

    print("</table>", file=fp)


def build_link_list(link_dict):
    parts = []
    for task_name, html_file in link_dict.items():
        parts.append(f'[<a href="{html_file}">{task_name}</a>]')
    return f"<p>links: {' '.join(parts)}</p>"


@click.command()
@click.argument('root')
def main(root):
    """The `render_model_vids.py` script will create a huge directory of videos
    for every available model & variant. This collects those videos into a
    series of HTML tables that you can view in your browser."""
    print("Building index")
    index = build_index(root)

    # figure out which files we're going to put data for each task in
    link_dict = OrderedDict()
    for task in sorted(index.keys()):
        link_dict[task] = task + ".html"
    link_list = build_link_list(link_dict)

    # build task-specific HTML files
    for task, task_dict in sorted(index.items()):
        print("Processing task", task)
        with open(os.path.join(root, link_dict[task]), "w") as fp:
            fp.write(PREAMBLE % ("results for " + task))
            print(f"<h1>Results on {task}</h1>", file=fp)
            print(link_list, file=fp)
            print('<p><a href="#" onclick="syncVids()">sync vids</a></p>',
                  file=fp)
            render_table(task, task_dict, fp)
            fp.write(ENDING)

    # also create an index file that links to each of those
    print("Writing index")
    with open(os.path.join(root, "index.html"), "w") as fp:
        fp.write(PREAMBLE % "results index")
        print("<h1>results index</h1>", file=fp)
        print(link_list, file=fp)
        fp.write(ENDING)

    print(f"Done, try `pushd {root} && python3 -m http.server; popd`")


if __name__ == '__main__':
    main()
controls
