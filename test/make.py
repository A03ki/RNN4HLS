import os
import subprocess


def run_make():  # build .so file by make command
    if not os.path.exists('build'):
        os.mkdir('build')
    proc = subprocess.run(["make"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    return proc.stdout.decode("utf8"), proc.stderr.decode("utf8")
