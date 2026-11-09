"""Verification Suite for Flight Domain Conversation and Anti-Repetition."""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.llm_brain import LLMBrain


async def test_flight_multi_turn_flow():
    print("\n" + "=" * 70)
    print(" [TEST] Multi-Turn Flight Conversation & Anti-Repetition Verification")
    print("=" * 70)

    llm = LLMBrain()

    # Turn 1: Initial query about Chandigarh to Lucknow flights
    t1_input = "Mai chandigarh se lucknow ki flights ke baare me dekhra tha"
    r1 = await llm.generate_reply(t1_input)
    print(f"\nUser 1: '{t1_input}'")
    print(f"Bot 1 : '{r1}'")

    assert "Ji bilkul, main samajh gaya" not in r1, "Turn 1 returned the generic unhelpful fallback!"
    assert any(w in r1.lower() for w in ("flight", "chandigarh", "lucknow")), "Turn 1 missing flight context!"

    # Turn 2: User says "accha"
    t2_input = "accha"
    r2 = await llm.generate_reply(t2_input)
    print(f"\nUser 2: '{t2_input}'")
    print(f"Bot 2 : '{r2}'")

    assert r2 != r1, f"Bot repeated the exact same answer on 'accha'! ({r2})"
    assert "Ji bilkul, main samajh gaya" not in r2, "Turn 2 returned the generic fallback!"
    assert any(w in r2.lower() for w in ("flight", "ghante", "timing", "fare", "chandigarh", "lucknow", "available")), "Turn 2 lost flight context!"

    # Turn 3: User says "haan"
    t3_input = "haan"
    r3 = await llm.generate_reply(t3_input)
    print(f"\nUser 3: '{t3_input}'")
    print(f"Bot 3 : '{r3}'")

    assert r3 != r2, f"Bot repeated the exact same answer on 'haan'! ({r3})"
    assert r3 != r1, f"Bot repeated Turn 1 answer on 'haan'!"
    assert "Ji bilkul, main samajh gaya" not in r3, "Turn 3 returned generic fallback!"

    # Turn 4: User asks follow-up about morning flight timing
    t4_input = "subah wali flight kitne baje hai?"
    r4 = await llm.generate_reply(t4_input)
    print(f"\nUser 4: '{t4_input}'")
    print(f"Bot 4 : '{r4}'")

    assert r4 != r3, "Bot repeated previous answer on timing query!"
    assert any(w in r4 for w in ("9:15", "subah", "baje")), f"Turn 4 failed to answer timing! Got: {r4}"

    # Turn 5: User asks about fare
    t5_input = "aur kiraya kitna hoga?"
    r5 = await llm.generate_reply(t5_input)
    print(f"\nUser 5: '{t5_input}'")
    print(f"Bot 5 : '{r5}'")

    assert r5 != r4, "Bot repeated previous answer on fare query!"
    assert any(w in r5 for w in ("fare", "rupaye", "kiraya", "4,200", "economy")), f"Turn 5 failed to answer fare! Got: {r5}"

    print("\n" + "=" * 70)
    print(" 🎉 ALL 5 MULTI-TURN FLIGHT CONVERSATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_flight_multi_turn_flow())
