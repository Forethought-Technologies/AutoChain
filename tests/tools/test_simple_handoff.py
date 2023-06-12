from minichain.tools.simple_handoff.tools import HandOffToAgent


def test_simple_handoff() -> None:
    handoff = HandOffToAgent()
    msg = handoff.run("")
    assert "hand you off" in msg
