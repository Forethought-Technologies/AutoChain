from langchain.agents import AgentType
from langchain.tools import Tool as LCTool

from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.utils import get_args
from autochain.workflows_evaluation.langchain_eval.langchain_test_utils import (
    create_langchain_from_test,
)


def search_restaurant(location: str, **kwargs):
    """Returns order information as a dictionary, where order_status can be "shipped" or "not_shipped" """
    return [
        {
            "restaurant_name": f"ABC dumplings",
            "food_type": "Chinese",
        },
        {
            "restaurant_name": f"KK sushi",
            "food_type": "Japanese",
        },
    ]


def get_menu(restaurant_name: str, **kwargs):
    """Changes the shipping address for unshipped orders. Requires the order_id and the new_address inputs"""
    if "dumpling" in restaurant_name.lower():
        return ["tan tan noodles", "mushroom fried rice", "pork buns"]
    elif "sushi" in restaurant_name.lower():
        return ["unagi roll", "tuna sushi", "fried tofu"]
    else:
        return "not found"


class TestFindFoodNearMeWithLC(BaseTest):
    prompt = """You are able to search restaurant and find corresponding food type for user. 
First, searching restaurants for users and responds to user with restaurants met user food preference.
Secondly, only if user requested, use tool to get menu. From menu list, responds to 
users with dishes they might like. 
If no restaurant met user requirements, replies with i don't know.
"""

    tools = [
        LCTool(
            name="search restaurant",
            func=search_restaurant,
            description="""This function searches all available restaurants and their food types
Input args: location""",
        ),
        LCTool(
            name="get menu",
            func=get_menu,
            description="""This function gets the name of all dishes for the restaurant
Input args: restaurant_name""",
        ),
    ]

    test_cases = [
        TestCase(
            test_name="find a chinese restaurant",
            user_context="find the name of the any chinese restaurant; you are located in new "
            "york city",
            expected_outcome="found ABC dumplings",
        ),
        TestCase(
            test_name="failed to find any french restaurant",
            user_context="find the name of the any french restaurant; you are located in new "
            "york city",
            expected_outcome="cannot find any french restaurants",
        ),
        TestCase(
            test_name="find vegetarian option for a Japanese restaurant",
            user_context="find a Japanese restaurant and all the vegetarian options; you are located in new "
            "york city",
            expected_outcome="found KK sushi and fired tofu",
        ),
    ]

    chain = create_langchain_from_test(
        tools=tools,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        prefix=prompt,
    )


if __name__ == "__main__":
    tests = WorkflowTester(
        tests=[TestFindFoodNearMeWithLC()], output_dir="./test_results"
    )

    args = get_args()
    if args.interact:
        tests.run_interactive()
    else:
        tests.run_all_tests()
