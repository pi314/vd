import argparse
import datetime
import sys
import threading
import time

from . import lib_subproc as subproc


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def worker(streams, cmd, delay, stop_signal):
    # Get initial content from command for excluding it
    p = subproc.run(cmd)
    if p.returncode:
        return

    old_lines = p.stdout.lines
    sponged_lines = []

    while True:
        if stop_signal.is_set():
            break

        p = subproc.run(cmd)

        if p.returncode:
            break

        new_lines = p.stdout.lines

        if old_lines != new_lines:
            for line in new_lines:
                sponged_lines.append(line)

                # Print timestamp for reference so you know what's going on
                if not sys.stdout.isatty() or True:
                    print_err('[' + str(datetime.datetime.now()) + ']', line)

            old_lines = new_lines

            subproc.run(['ntfy', '-t', 'Copied', '{} lines'.format(len(sponged_lines))])

        if cmd[0] != 'sleep':
            time.sleep(delay)

    streams[1].writelines(sponged_lines)


def main():
    try:
        parser = argparse.ArgumentParser(prog='sponge',
                description='sponge',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-d', '--delay', default=0.2, type=float, help='Delay in seconds')
        parser.add_argument('command', nargs='*', default=None, help='Command to run')

        args = parser.parse_args()

        if not args.command:
            sponged_lines = [line.rstrip() for line in sys.stdin.readlines()]
            for line in sponged_lines:
                print(line)
            sys.exit()

        stop_signal = threading.Event()

        cmd = subproc.run([worker, args.command, args.delay, stop_signal], stderr=None, wait=False)

        for line in sys.stdin:
            print('[ignored]', line.rstrip())
        stop_signal.set()

        cmd.wait()

        for line in cmd.stdout:
            print(line)

    except KeyboardInterrupt:
        print_err('KeyboardInterrupt')
