class OutputParserException(Exception):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.
    """

    pass


class ToolRunningError(Exception):
    """Exception when tool fails to run"""

    def __init__(self, message):
        self.message = message
