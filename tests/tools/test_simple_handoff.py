from autochain.tools.simple_handoff.tool import HandOffToAgent


def test_simple_handoff() -> None:
    handoff = HandOffToAgent()
    msg = handoff.run()
    assert handoff.handoff_msg == msg
