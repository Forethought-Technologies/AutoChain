import argparse

from minichain.tools.base import Tool
from minichain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from minichain.workflows_evaluation.test_utils import get_test_args


class TestExchangeOrReturnTest(BaseTest):

    @staticmethod
    def check_return_eligibility(order_number: str, **kwargs):
        if order_number == "123" or order_number == "345":
            return {
                "status_code": 200, "return_ok": True
            }
        else:
            return {
                "status_code": 200, "return_ok": False
            }

    @staticmethod
    def check_user_status(email: str):
        if "test@gmail.com" in email:
            return {
                "status_code": 200, "user_location": "international", "order_number": "123"
            }
        else:
            return {
                "status_code": 200, "user_location": "domestic", "order_number": "345"
            }

    policy = """"Assist customers with returns and exchanges.
    1. Verify the customer's order number and check if the items have been worn, washed, or if the tags are attached.
    2. If the items are eligible for return or exchange, guide the customer through the return process by providing a link to the return form.
    3. Offer the option of using a Happy Returns QR code or a prepaid FedEx return label for returning the items.
    4. If the customer needs to cancel or modify a return form, assist them in doing so and provide guidance on how to redo the process correctly.
    5. For international customers, inform them that direct exchanges are not available and suggest setting up a return for a refund and creating a new order for the desired items.
    6. In case of final sale items, evaluate the situation and consider making a one-time exception for an exchange or return if necessary.
    7. Ensure the customer is aware of any discounts or shipping fees associated with their new order and provide an invoice if needed."
    """

    tools = [
        Tool(
            name="check return eligibility",
            func=check_return_eligibility,
            description="""This function if order can be returned or exchanged
Input args: order_number: non-empty str
Output values: status_code: int, return_ok: bool """
        ),
        Tool(
            name="change user status",
            func=check_user_status,
            description="""This function checks user location and determine if user is 
international or domestic based on email
Input args: email: non-empty str
Output values: status_code: int, user_location: str, order_number: str"""
        ),
    ]

    test_cases = [
        TestCase(test_name="exchange with ok order number and email",
                 user_query="i would like to return my item",
                 user_context="order number is 345, email is blah@gmail.com",
                 expected_outcome="issue return to user"),
        TestCase(test_name="exchange is not ok based on order number",
                 user_query="i would like to return my item",
                 user_context="Your order id is 638; you name is Jacky; email is blah@gmail.com",
                 expected_outcome="cannot issue refund because user is international"),
        # TestCase(test_name="cannot issue refund because user is international",
        #          user_query="i would like to return my item",
        #          user_context="Rou name is Jacky; email is blah@gmail.com, not sure about order "
        #                       "number",
        #          expected_outcome="cannot issue refund because user is international"),
    ]


if __name__ == '__main__':
    tests = WorkflowTester(tests=[TestExchangeOrReturnTest()], output_dir="./test_results")
    args = get_test_args()
    if args.interact:
        tests.run_interactive()
    else:
        tests.run_all_tests()
