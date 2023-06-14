from autochain.tools.base import Tool
from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.workflows_evaluation.test_utils import (
    get_test_args,
    create_chain_from_test,
)


class TestOrderStatusAndRefundRequest(BaseTest):
    @staticmethod
    def snowflake_order_status(order_id):
        if "6381" in order_id:
            return str(
                {
                    "status_code": 200,
                    "tracking_link": "https://www.fedex.com/fedextrack/no-results-found?trknbr=12312312321",
                    "carrier": "FedEx",
                    "location": "Minneapolis shipping center",
                    "message": "Expected arrival 12PM-8PM April 15th, 2023",
                    "last_time": "8:23AM, April 11th, 2023",
                }
            )
        else:
            return str({"status_code": 404, "message": "Order ID not found."})

    @staticmethod
    def validate_order_status_input(order_id):
        if order_id.isalnum():
            return "True"
        return "False"

    policy = """AI is responsible for the following policy
Policy description: When a customer reports that they have not received their order, first check the order status in the system. 
If the order has shipped, provide them with a tracking link and ask them to confirm if there was a \"Missed Delivery\" notice left behind. 
Request that they check around their property, with neighbors, and at their leasing office for any misplaced deliveries. 
Confirm the correct shipping address with the customer. 
If the order is still being processed or delayed, inform them of any expected shipping timeframes and assure them that they will receive an email confirmation with tracking information once it ships.
In case of lost or missing orders after all attempts to locate it have been exhausted, offer either a refund or replacement for their order. If opting for a replacement, require a signature for delivery and inform customers about rerouting options to local FedEx offices if needed."
"""

    tools = [
        Tool(
            name="Order Status",
            func=snowflake_order_status,
            description="""This function checks order status for a given order id.
Input args: order_id: non-empty str
Output values: status_code: int, order_id: str, tracking_url: str, message: str""",
        ),
        Tool(
            name="Validate Order Status",
            func=validate_order_status_input,
            description="""This function checks if the input order ID is alphanumeric and returns a boolean value.
Input args: order_id: non-empty str
Output values: is_order_valid: bool""",
        ),
    ]

    test_cases = [
        TestCase(
            test_name="get order success case",
            user_query="Where is my order",
            user_context="Your order id is 6381; you name is Jacky; email is jack@gmail.com",
            expected_outcome="retrieve order status and get shipping info",
        ),
        TestCase(
            test_name="cannot get order status",
            user_query="Where is my order",
            user_context="Your order id is 123; you name is Jacky; email is jack@gmail.com",
            expected_outcome="No order information retrieved, clarify order id and hand "
            "off to agent",
        ),
        TestCase(
            test_name="order lost and request refund",
            user_query="Where is my order",
            user_context="Your order id is 6381; you name is Jacky; Cannot "
            "find item anywhere and it is lost. I need refund for this order",
            expected_outcome="get refund or replacement",
        ),
    ]

    chain = create_chain_from_test(tools=tools, policy=policy)


if __name__ == "__main__":
    tester = WorkflowTester(
        tests=[TestOrderStatusAndRefundRequest()], output_dir="./test_results"
    )

    args = get_test_args()
    if args.interact:
        tester.run_interactive()
    else:
        tester.run_all_tests()
