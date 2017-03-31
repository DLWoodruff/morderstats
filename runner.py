"""
runner.py
  This program will run python programs that have attached configuration (.txt) files.
  The text file should start with the line "command/exec program.py", where program.py stands for the python file
  you want to run and a list of all the options, each in their own line, to run the program with.
  The program ignores comments marked with the "#" symbol.
"""

import sys
import os
import subprocess


def parse_commands(filename):
    """Accepts a configuration filename and
    parses out the program name and the options
    from the file and returns them.
    """
    with open(filename) as f:
        for line in f:
            if line.startswith("command/exec"):
                _, program = line.split()
                break
        else:
            print("No Program name found")
            print("Include a line that starts with 'command/exec'")
            print("followed by the program name")
            sys.exit()

        for line in f:
            if line.startswith('--'):
                line = line.replace('=', ' ')
                line = line.replace('|', os.sep)
                options.extend(line.split())
    return program, options


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("You must list the file with the program configurations")
        print("after the program name")
        print("Usage: python runner.py config_file")
        sys.exit()

    config_filename = sys.argv[1]
    if not(os.path.isfile(config_filename)):
        print("{} is not a file or does not exist".format(config_filename))
    options = []
    program, options = parse_commands(config_filename)
    subprocess.call(['python'] + [program] + options)
