#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"


# Print iterations progress
def printprogress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    white = ' ' * 50
    print(f'\r{prefix} |{bar}| {percent}% {white}', end = printEnd)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def finish(total=100, prefix='', suffix=''):
    # Force stop the progress bar
    print(end="")