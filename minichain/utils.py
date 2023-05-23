from colorama import Style


def print_with_color(text: str, color: str):
    print(color + text)
    print(Style.RESET_ALL)
