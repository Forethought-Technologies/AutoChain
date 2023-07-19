from autochain.tools.base import Tool
from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.workflows_evaluation.test_utils import (
    create_chain_from_test,
)
from autochain.utils import get_args


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
        }
    ]


def get_menu(restaurant_name: str, **kwargs):
    """Changes the shipping address for unshipped orders. Requires the order_id and the new_address inputs"""
    if "dumpling" in restaurant_name.lower():
        return ["tan tan noodles", "mushroom fried rice", "pork buns"]
    elif "sushi" in restaurant_name.lower():
        return ["unagi roll", "tuna sushi", "fried tofu"]
    else:
        return "not found"


class TestFindFoodNearMe(BaseTest):
    prompt = """You are a restaurant planner for user. 
Searching restaurants for users. 
Only show user restaurants met user food type preference. 
Only if user requested, find restaurant's menu and recommend dishes user might like. 
"""

    tools = [
        Tool(
            func=search_restaurant,
            description="""This function searches all available restaurants
Input args: location""",
        ),
        Tool(
            func=get_menu,
            description="""This function gets the name of all dishes for the restaurant
Input args: restaurant_name""",
        ),
    ]

    test_cases = [
        # TestCase(
        #     test_name="find a chinese restaurant",
        #     user_context="find the name of the any chinese restaurant",
        #     expected_outcome="found ABC dumplings",
        # ),
        # TestCase(
        #     test_name="failed to find any french restaurant",
        #     user_context="find the name of the any french restaurant",
        #     expected_outcome="cannot find any french restaurants",
        # ),
        TestCase(
            test_name="find vegetarian option for a Japanese restaurant",
            user_context="find a Japanese restaurant and all the vegetarian options",
            expected_outcome="found KK sushi and fired tofu",
        ),
    ]

    chain = create_chain_from_test(tools=tools, prompt=prompt)


if __name__ == "__main__":
    tester = WorkflowTester(
        tests=[TestFindFoodNearMe()], output_dir="./test_results"
    )

    args = get_args()
    if args.interact:
        tester.run_interactive()
    else:
        tester.run_all_tests()
