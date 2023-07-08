from autochain.tools.base import Tool
from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.workflows_evaluation.test_utils import (
    create_chain_from_test,
)
from autochain.utils import get_args


def check_order_status(order_id: str, **kwargs):
    """Returns order information as a dictionary, where order_status can be "shipped" or "not_shipped" """
    if order_id == "123":
        return {
            "status_code": 200,
            "order_id": "123",
            "order_status": "shipped",
            "tracking_url": "example.com/123",
            "shipping_address": "123 ivy street san francisco ca",
        }
    elif order_id == "456":
        return {
            "status_code": 200,
            "order_id": "456",
            "order_status": "not_shipped",
            "tracking_url": "example.com/456",
            "shipping_address": "234 spear street san francisco ca",
        }
    else:
        return {"status_code": 400, "message": "order not found"}


def change_shipping_address(order_id: str, new_address: str, **kwargs):
    """Changes the shipping address for unshipped orders. Requires the order_id and the new_address inputs"""
    return {
        "status_code": 200,
        "order_id": order_id,
        "shipping_address": new_address,
    }


class TestChangeShippingAddress(BaseTest):
    prompt = """You are an AI customer support assistant who tries to help with shipping 
address questions.
When a customer requests to change their shipping address, verify the order status in the system based on order id.
If the order has not yet shipped, update the shipping address as requested and confirm with the customer that it has been updated. 
If the order has already shipped, inform them that it is not possible to change the shipping address at this stage and provide assistance on how to proceed with exchanges, by following instructions at example.com/returns.
"""

    tools = [
        Tool(
            func=check_order_status,
            description="""This function checks the order status based on order_id
Input args: order_id: non-empty str""",
        ),
        Tool(
            func=change_shipping_address,
            description="""This function change the shipping address based on provided 
order_id and new_address 
Input args: order_id: non-empty str, new_address: non-empty str""",
        ),
    ]

    test_cases = [
        TestCase(
            test_name="change shipping address",
            user_context="want to change shipping address; order id is 456. the new address is "
            "234 spear st, san francisco",
            expected_outcome="found order status and changed shipping address",
        ),
        TestCase(
            test_name="failed changing shipping address, no order id",
            user_context="want to change shipping address; don't know about order id. the new "
            "address is 234 spear st, san francisco",
            expected_outcome="cannot find the order status, failed to change shipping "
            "address",
        ),
        TestCase(
            test_name="failed changing shipping address, shipped item",
            user_context="want to change shipping address; order id is 123. the new address is 234 spear st, "
            "san francisco",
            expected_outcome="inform user cannot change shipping address and hand off to "
            "agent",
        ),
    ]

    chain = create_chain_from_test(tools=tools, prompt=prompt)


if __name__ == "__main__":
    tester = WorkflowTester(
        tests=[TestChangeShippingAddress()], output_dir="./test_results"
    )

    args = get_args()
    if args.interact:
        tester.run_interactive()
    else:
        tester.run_all_tests()
