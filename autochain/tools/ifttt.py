"""From https://github.com/SidU/teams-langchain-js/wiki/Connecting-IFTTT-Services.

# Creating a webhook
- Go to https://ifttt.com/create

# Configuring the "If This"
- Click on the "If This" button in the IFTTT interface.
- Search for "Webhooks" in the search bar.
- Choose the first option for "Receive a web request with a JSON payload."
- Choose an Event Name that is specific to the service you plan to connect to.
This will make it easier for you to manage the webhook URL.
For example, if you're connecting to Spotify, you could use "Spotify" as your
Event Name.
- Click the "Create Trigger" button to save your settings and create your webhook.

# Configuring the "Then That"
- Tap on the "Then That" button in the IFTTT interface.
- Search for the service you want to connect, such as Spotify.
- Choose an action from the service, such as "Add track to a playlist".
- Configure the action by specifying the necessary details, such as the playlist name,
e.g., "Songs from AI".
- Reference the JSON Payload received by the Webhook in your action. For the Spotify
scenario, choose "{{JsonPayload}}" as your search query.
- Tap the "Create Action" button to save your action settings.
- Once you have finished configuring your action, click the "Finish" button to
complete the setup.
- Congratulations! You have successfully connected the Webhook to the desired
service, and you're ready to start receiving data and triggering actions 🎉

# Finishing up
- To get your webhook URL go to https://ifttt.com/maker_webhooks/settings
- Copy the IFTTT key value from there. The URL is of the form
https://maker.ifttt.com/use/YOUR_IFTTT_KEY. Grab the YOUR_IFTTT_KEY value.
"""


from typing import Any

from autochain.tools.base import Tool

import requests


class IFTTTWebhook(Tool):
    """IFTTT Webhook.

    Args:
        name: name of the tool
        description: description of the tool
        url: url to hit with the json event.
    """
    
    name = "IFTTT Webhook"
    description = "Tool for connecting IFTTT services"
    url: str

    def _run(self, input_data:str) -> str:
        body = {"this": input_data}
        response = requests.post(self.url, data=body)
        return response.text
