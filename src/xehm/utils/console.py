# Print iterations progress - designed to be called in a loop with no interleaved printing
# A nice fill character is █ (ascii: 219)
def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1,
                       length: int = 80, fill: str = '█', newline_end: bool = True):
    # TODO: 'f' string this
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}%% {suffix}', end="", flush=True)
    # Blank print on Complete to clear the line
    if iteration == total and newline_end:
        print()
