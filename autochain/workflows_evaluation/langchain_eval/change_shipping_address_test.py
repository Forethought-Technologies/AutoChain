from langchain.agents import AgentType
from langchain.tools import Tool as LCTool

from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.utils import get_args
from autochain.workflows_evaluation.langchain_eval.langchain_test_utils import (
    create_langchain_from_test,
)


def check_order_status(order_id: str, **kwargs):
    """Returns order information as a dictionary, where order_status can be "shipped" or "not_shipped" """
    if "123" in order_id:
        return {
            "status_code": 200,
            "order_id": "123",
            "order_status": "shipped",
            "tracking_url": "example.com/123",
            "shipping_address": "123 ivy street san francisco ca",
        }
    elif "456" in order_id:
        return {
            "status_code": 200,
            "order_id": "456",
            "order_status": "not_shipped",
            "tracking_url": "example.com/456",
            "shipping_address": "234 spear street san francisco ca",
        }
    else:
        return {"status_code": 400, "message": "order not found"}


def change_shipping_address(order_id: str, new_address: str = "", **kwargs):
    """Changes the shipping address for unshipped orders. Requires the order_id and the new_address inputs"""
    return {
        "status_code": 200,
        "order_id": order_id,
        "shipping_address": new_address,
    }


class TestChangeShippingAddressWithLC(BaseTest):
    prompt = """You are an AI assistant for customer support who tries to help with shipping 
address questions.
When a customer requests to change their shipping address, verify the order status in the system based on order id.
If the order has not yet shipped, update the shipping address as requested and confirm with the customer that it has been updated. 
If the order has already shipped, inform them that it is not possible to change the shipping address at this stage and provide assistance on how to proceed with exchanges, by following instructions at example.com/returns.

TOOLS:
------

Assistant has access to the following tools:
"""

    tools = [
        LCTool(
            name="check order status",
            func=check_order_status,
            description="""This function checks the order status based on order_id
Input args: order_id: non-empty str""",
        ),
        LCTool(
            name="change shipping address",
            func=change_shipping_address,
            description="""This function change the shipping address based on provided 
order_id and new_address 
Input args: order_id: non-empty str, new_address: non-empty str""",
        ),
    ]

    test_cases = [
        TestCase(
            test_name="change shipping address",
            user_context="order id is 456. the new address is 234 spear st, "
            "san francisco",
            expected_outcome="found order status and changed shipping address",
        ),
    ]

    chain = create_langchain_from_test(
        tools=tools,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        prefix=prompt,
    )


if __name__ == "__main__":
    tests = WorkflowTester(
        tests=[TestChangeShippingAddressWithLC()], output_dir="./test_results"
    )

    args = get_args()
    if args.interact:
        tests.run_interactive()
    else:
        tests.run_all_tests()
