import argparse
import logging
import os
from typing import Optional, Dict, Any

from colorama import Style


def print_with_color(text: str, color: str):
    if os.getenv("NO_COLOR"):
        print(text)
    else:
        print(color + text)
        print(Style.RESET_ALL)


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


def get_args():
    """Adding arguments for running test interactively or setting verbosity"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interact",
        "-i",
        action="store_true",
        help="if run interactively",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if show detailed contents, such as intermediate results and prompts",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    return args
