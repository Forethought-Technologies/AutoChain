from minichain.tools.base import Tool
from minichain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from minichain.workflows_evaluation.test_utils import get_test_args, create_chain_from_test


class TestImproperlyParkedVehicle(BaseTest):
    @staticmethod
    def check_active_reservation(user_id: str):
        if user_id in ["jack"]:
            return str({'status_code': 200, 'is_active_reservation': True, 'vehicle_id': 123})
        else:
            return str({'status_code': 200, 'is_active_reservation': False})

    @staticmethod
    def check_vehicle_parking(vehicle_id: int):
        """
        Checks if the vehicle is properly parked, requires the vehicle_id
        """
        if vehicle_id in [123, 456]:
            return str({'status_code': 200, 'illegally_parked': True})
        elif vehicle_id in [789]:
            return str({'status_code': 200, 'illegally_parked': False})
        else:
            return str({'status_code': 404, 'message': 'vehicle was not found'})

    policy = """Assistant helps the customer figure out whether the vehicle in question is properly parked.
Assistant can ask the customer for their user_id or the vehicle_id.  
This might be a vehicle the customer used, or just someone else's vehicle that they are reporting. Assistant should check if the customer currently has an active rental.
Note that if the vehicle is not found in our system Assistant must escalate to customer support representative after checking if the user is a law enforcement officer.
"""
    tools = [
        Tool(
            name="check user active reservation",
            func=check_active_reservation,
            description="""This function checks if user has active reservation and vehicle id
Input args: user_id: non-empty str
Output values: is_active_reservation: bool, vehicle_id: int"""
        ),
        Tool(
            name="check vehicle parking status",
            func=check_vehicle_parking,
            description="""This function checks vehicle status using vehicle id.
Input args: vehicle_id: non-empty int
Output values: illegally_parked: bool, message: str"""
        ),
    ]

    test_cases = [
        TestCase(test_name="found name and resolved",
                 user_query="I parked a Lime vehicle on the street, can you tell me if it's "
                            "properly parked?",
                 user_context="user id is jack. not sure about vehicle id or plate number",
                 expected_outcome="found active reservation and explain why it is not properly "
                                  "parked"),
        TestCase(test_name="no id provided, hand off to agent",
                 user_query="I parked a Lime vehicle on the street, can you tell me if it's "
                            "properly parked?",
                 user_context="don't know customer id or vehicle id, it is a green bike",
                 expected_outcome="cannot find active reservation, hand off to agent"),
        TestCase(test_name="wrong intent",
                 user_query="I tried to reserve a Lime vehicle but there weren't any available in my area.",
                 user_context="don't know customer id or vehicle id. no reservation",
                 expected_outcome="hand off to agent"),
        TestCase(test_name="check status of vehicle without id",
                 user_query="I want to know why there is a bike parked in my driveway",
                 user_context="don't know customer id or vehicle id. no reservation",
                 expected_outcome="hand off to agent, don't return fake vehicle status"),
    ]


if __name__ == '__main__':
    test = TestImproperlyParkedVehicle()
    chain = create_chain_from_test(test=test, policy=test.policy)
    tester = WorkflowTester(tests=[test], agent_chain=chain, output_dir="./test_results")
    args = get_test_args()
    if args.interact:
        tester.run_interactive()
    else:
        tester.run_all_tests()
