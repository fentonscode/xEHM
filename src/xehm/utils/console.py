from textwrap import fill

__all__ = ["print_separator_line", "print_progress_bar", "print_header"]


# Print a separator line on the console
def print_separator_line(char: str = "-", length: int = 80):
    print(char * length)


def print_header(msg: str, separator_char: str = "-", line_length: int = 80):
    lines = fill(msg, width = line_length)
    print_separator_line(separator_char, line_length)
    print(lines)
    print_separator_line(separator_char, line_length)


# Print iterations progress - designed to be called in a loop with no interleaved printing
# A nice fill character is █ (ascii: 219)
def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1,
                       length: int = 80, fill: str = '█', newline_end: bool = True):
    p_value = 100 * (iteration / float(total))
    percent = f"{p_value:.{decimals}}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}%% {suffix}', end="", flush=True)
    # Blank print on Complete to clear the line
    if iteration == total and newline_end:
        print()
