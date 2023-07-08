from langchain.agents import AgentType
from langchain.tools import Tool as LCTool

from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.utils import get_args
from autochain.workflows_evaluation.langchain_eval.langchain_test_utils import (
    create_langchain_from_test,
)


def get_item_spec(item_name: str, **kwargs):
    if "toy" in item_name.lower():
        return {"name": "toy bear", "color": "red", "age_group": "1-5 years old"}
    elif "printer" in item_name.lower():
        return {
            "name": "Wireless Printer",
            "printer_type": "Printer, Scanner, Copier",
            "color_print_speed": "5.5 page per minute",
            "mono_print_speed": "7.5 page per minute",
        }
    else:
        return {}


def search_image_path_for_item(item_name: str):
    if "toy" in item_name.lower():
        return "images/toy.png"
    elif "printer" in item_name.lower():
        return "images/awesome_printer.png"
    else:
        return ""


class TestGenerateAdsWithLC(BaseTest):
    prompt = """Your goals is helping user to generate an advertisement for user requested 
product and find relevant image path for the item.
You would first clarify what product you would write advertisement for and what are the key 
points should be included in the ads.
Based on item name, you could get its specifications that can be used in advertisement.
Then, you need to search and include an image path for the item at the bottom of advertisement. 
You could find relevant images path with tool provided and search of relevant image using query.
Generate advertisement with image path.

TOOLS:
------

Assistant has access to the following tools:
"""

    tools = [
        LCTool(
            name="get item spec",
            func=get_item_spec,
            description="""This function get item spec by searching for item name
Input args: item_name: non-empty str""",
        ),
        LCTool(
            name="search image path for item",
            func=search_image_path_for_item,
            description="""This function retrieves relevant image path for a given search query
Input args: item_name: str""",
        ),
    ]

    test_cases = [
        TestCase(
            test_name="ads for toy bear",
            user_context="Write me an advertisement for toy bear; item name is 'toy bear'. it is "
            "cute and made in USA, they should be "
            "included in the ads. Ads should include image",
            expected_outcome="generate an advertisement for toy bear and mentions it is cute. "
            "Also ads should include an image path",
        ),
        TestCase(
            test_name="printer ads",
            user_context="write me an advertisement for printer; item name is 'good printer'. "
            "printer is used and in good condition. "
            "Ads should include image",
            expected_outcome="generate an advertisement for wireless printer and mentions it is "
            "wireless, can be used as scanner and is used. Also ads should "
            "include an image path",
        ),
    ]

    chain = create_langchain_from_test(
        tools=tools,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        prefix=prompt,
    )


if __name__ == "__main__":
    tests = WorkflowTester(tests=[TestGenerateAdsWithLC()], output_dir="./test_results")

    args = get_args()
    if args.interact:
        tests.run_interactive()
    else:
        tests.run_all_tests()
