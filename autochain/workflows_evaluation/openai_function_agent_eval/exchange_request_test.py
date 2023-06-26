from autochain.agent.openai_funtions_agent.openai_functions_agent import (
    OpenAIFunctionsAgent,
)

from autochain.models.chat_openai import ChatOpenAI

from autochain.tools.base import Tool
from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.workflows_evaluation.test_utils import (
    get_test_args,
    create_chain_from_test,
)


def check_return_eligibility(order_number: str, **kwargs):
    if order_number == "123" or order_number == "345":
        return {"status_code": 200, "return_ok": True}
    else:
        return {"status_code": 200, "return_ok": False}


def check_user_status(email: str):
    if "test@gmail.com" in email:
        return {
            "status_code": 200,
            "user_location": "international",
            "order_number": "123",
        }
    else:
        return {
            "status_code": 200,
            "user_location": "domestic",
            "order_number": "345",
        }


class TestExchangeOrReturnTest(BaseTest):
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
            func=check_return_eligibility,
            description="""This function if order can be returned or exchanged
Input args: order_number: non-empty str
Output values: status_code: int, return_ok: bool """,
        ),
        Tool(
            func=check_user_status,
            description="""This function checks user location and determine if user is 
international or domestic based on email
Input args: email: non-empty str
Output values: status_code: int, user_location: str, order_number: str""",
        ),
    ]

    test_cases = [
        TestCase(
            test_name="exchange with ok order number and email",
            user_query="i would like to return my item",
            user_context="order number is 345, email is blah@gmail.com",
            expected_outcome="issue return to user",
        ),
        TestCase(
            test_name="exchange is not ok based on order number",
            user_query="i would like to return my item",
            user_context="Your order id is 638; you name is Jacky; email is blah@gmail.com",
            expected_outcome="cannot issue refund because user is international",
        ),
    ]

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
    chain = create_chain_from_test(
        tools=tools, agent_cls=OpenAIFunctionsAgent, llm=llm, prompt=policy
    )


if __name__ == "__main__":
    tester = WorkflowTester(
        tests=[TestExchangeOrReturnTest()], output_dir="./test_results"
    )

    args = get_test_args()
    if args.interact:
        tester.run_interactive()
    else:
        tester.run_all_tests()
