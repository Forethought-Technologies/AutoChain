
from minichain.agent.support_agent import SupportAgent
from minichain.chain.chain import DefaultChain
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.chat_openai import ChatOpenAI
from minichain.tools.base import Tool

policy = """AI is responsible for the following policy
Policy description: When a customer reports that they have not received their order, first check the order status in the system. 
If the order has shipped, provide them with a tracking link and ask them to confirm if there was a \"Missed Delivery\" notice left behind. 
Request that they check around their property, with neighbors, and at their leasing office for any misplaced deliveries. 
Confirm the correct shipping address with the customer. 
If the order is still being processed or delayed, inform them of any expected shipping timeframes and assure them that they will receive an email confirmation with tracking information once it ships.
In case of lost or missing orders after all attempts to locate it have been exhausted, offer either a refund or replacement for their order. If opting for a replacement, require a signature for delivery and inform customers about rerouting options to local FedEx offices if needed."
"""


def snowflake_order_status(order_id):
    if "6381" in order_id:
        return str({
            "status_code": 200,
            'tracking_link': "https://www.fedex.com/fedextrack/no-results-found?trknbr=12312312321",
            'carrier': "FedEx",
            'location': 'Minneapolis shipping center',
            'message': 'Expected arrival 12PM-8PM April 15th, 2023',
            'last_time': '8:23AM, April 11th, 2023',
        })
    else:
        return str({
            'status_code': 404,
            'message': 'Order ID not found.'
        })


def validate_order_status_input(order_id):
    if order_id.isalnum():
        return "True"
    return "False"


tools = [
    Tool(
        name="Order Status",
        func=snowflake_order_status,
        description="""This function checks order status for a given order id.
                Args: order_id"""
    ),
    Tool(
        name="Validate Order Status",
        func=validate_order_status_input,
        description="""This function checks if the input order ID is alphanumeric and returns a boolean value.
                Args: order_id"""
    ),
]

llm = ChatOpenAI(temperature=0)
agent = SupportAgent.from_llm_and_tools(llm, tools, policy_desp=policy)
memory = BufferMemory()

chain = DefaultChain(tools=tools, agent=agent, memory=memory)
response = chain.run("where is my order", return_only_outputs=True)
while True:
    print(f">>> Assistant: {response['output']}")
    print("\n")
    query = input(">>> User: ")
    response = chain.run(query)
