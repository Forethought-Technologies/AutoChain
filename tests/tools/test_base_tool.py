import pytest

from autochain.tools.base import Tool


def sample_tool_func(k, *arg, **kwargs):
    return f"run with {k}"


def test_run_tool():
    tool = Tool(
        func=sample_tool_func,
        description="""This is just a dummy tool""",
    )

    output = tool.run("test")
    assert output == "run with test"


def test_tool_name_override():
    new_test_name = "new_name"
    tool = Tool(
        name=new_test_name,
        func=sample_tool_func,
        description="""This is just a dummy tool""",
    )

    assert tool.name == new_test_name


def test_arg_description():
    valid_arg_description = {"k": "key of the arg"}

    invalid_arg_description = {"not_k": "key of the arg"}

    _ = Tool(
        func=sample_tool_func,
        description="""This is just a dummy tool""",
        arg_description=valid_arg_description,
    )

    with pytest.raises(ValueError):
        _ = Tool(
            func=sample_tool_func,
            description="""This is just a dummy tool""",
            arg_description=invalid_arg_description,
        )
