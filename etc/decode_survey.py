#!/usr/bin/env python3
import base64
import json

import click


def sanitise_arr(arr):
    out_dict = {}
    for elem in arr:
        name = elem['name']
        value = elem['value']
        name_parts = name.split(':')
        d = out_dict
        for p in name_parts[:-1]:
            d = d.setdefault(p, {})
        d[name_parts[-1]] = value
    return out_dict


@click.command()
@click.argument('fp', type=click.File('r'))
def main(fp):
    decode_dict = {}
    responses_by_task = {}
    for line in fp:
        line = line.strip()
        if not line:
            continue
        name, rest = line.split(':')
        rest = rest.strip()
        un_b64 = base64.b64decode(rest)
        json_arr = json.loads(un_b64)
        sane = sanitise_arr(json_arr)
        decode_dict[name] = sane

    for person_name, tasks_dict in decode_dict.items():
        for task_num, task_resp in tasks_dict['tasks'].items():
            env_name = task_resp['env-name']
            task_dict = responses_by_task.setdefault(env_name, {})
            task_dict['num'] = task_num
            task_dict.setdefault('demo',
                                 {})[person_name] = task_resp['demo-desc']
            for var_num, desc in task_resp['eval-descs'].items():
                task_dict.setdefault('test-' + var_num, {})[person_name] = desc

    for env_name, resp_dict in sorted(responses_by_task.items(),
                                      key=lambda t: int(t[1]['num'])):
        num = resp_dict['num']
        print(f'### Responses for environment "{env_name}" (task {num})')
        print()
        for key, values in sorted(resp_dict.items()):
            if key == 'num':
                continue
            print(f'#### Descriptions of {key} configuration')
            print()
            for person, resp in sorted(values.items()):
                # world's worst inline text escaping
                resp = resp.strip().replace('<', '&lt;') \
                       .replace('> ', '&gt;') \
                       .replace('\n', '<br/>')
                print(f'- {person}: {resp}')
            print()


if __name__ == '__main__':
    main()
