from minichain.workflows_evaluation.base_test import WorkflowTester
from minichain.workflows_evaluation.change_shipping_address_test import TestChangeShippingAddress
from minichain.workflows_evaluation.exchange_request_test import TestExchangeOrReturnTest
from minichain.workflows_evaluation.improperly_parked_vehicle_test import \
    TestImproperlyParkedVehicle
from minichain.workflows_evaluation.order_status_request_test import \
    TestOrderStatusAndRefundRequest

if __name__ == '__main__':
    tests = WorkflowTester(tests=[TestChangeShippingAddress(),
                                  TestOrderStatusAndRefundRequest(),
                                  TestImproperlyParkedVehicle(),
                                  TestExchangeOrReturnTest()],
                           output_dir="./test_results")
    tests.run_all_tests()
