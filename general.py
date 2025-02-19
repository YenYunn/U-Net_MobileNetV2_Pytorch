import inspect
import logging
import os
from pathlib import Path
from typing import Optional


def set_logging(name=None):
    level = logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(level)
    log.addHandler(handler)


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


set_logging()
LOGGER = logging.getLogger('Unet')

def print_args(args: Optional[dict] = None, show_file=True, show_fcn=False):
    x = inspect.currentframe().f_back
    file, _, fcn, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}

    title = f'{Path(file).stem}: {fcn}' if show_file or show_fcn else 'Function Arguments'
    title_str = colorstr('cyan', 'bold', title)
    separator = colorstr('bright_black', '=' * 40)

    max_key_length = max(len(k) for k in args.keys()) if args else 0

    args_str = '\n'.join(f"{colorstr('yellow', k.ljust(max_key_length))} = {colorstr('green', v)}" for k, v in args.items())

    output = f'\n{separator}\n{title_str}\n{separator}\n{args_str}\n{separator}'
    LOGGER.info(output)


# 舊有格式，全部擠在同一行
# def print_args(args: Optional[dict] = None, show_file=True, show_fcn=False):
#     x = inspect.currentframe().f_back
#     file, _, fcn, _, _ = inspect.getframeinfo(x)
#     if args is None:
#         args, _, _, frm = inspect.getargvalues(x)
#         args = {k: v for k, v in frm.items() if k in args}
# 
#     s = (f'{Path(file).stem}: ' if show_file else '') + (f'{fcn}: ' if show_fcn else '')
#     args_str = '\n'.join(f'{k}={v}' for k, v in args.items())
# 
#     LOGGER.info(colorstr(s) + '\n' + args_str)
