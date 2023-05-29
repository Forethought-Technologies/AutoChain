from tests.workflow_tests.base_test import WorkflowTester
from tests.workflow_tests.change_shipping_address_test import TestChangeShippingAddress
from tests.workflow_tests.improperly_parked_vehicle_test import TestImproperlyParkedVehicle
from tests.workflow_tests.refund_request_test import TestOrderStatusAndRefundRequest

if __name__ == '__main__':
    tests = WorkflowTester(tests=[TestChangeShippingAddress(),
                                  TestOrderStatusAndRefundRequest(),
                                  TestImproperlyParkedVehicle()],
                           output_dir="./test_results")
    tests.run_all_tests()
