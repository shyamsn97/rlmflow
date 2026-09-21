from pydantic import JsonValue, TypeAdapter

from examples.shepherd.sokoban import Sokoban

BOARD = [
    "############",
    "#  .   .  ##",
    "### # # # ##",
    "#@$        #",
    "#   # # # .#",
    "#  $     $ #",
    "#  ## # #  #",
    "# .     .  #",
    "############",
]


def test_sokoban_publishes_strict_json_values():
    env = {}
    game = Sokoban(BOARD, env=env)

    TypeAdapter(dict[str, JsonValue]).validate_python(env, strict=True)
    game.push("right")
    TypeAdapter(dict[str, JsonValue]).validate_python(env, strict=True)

    assert env["player"] == [3, 2]
    assert all(isinstance(position, list) for position in env["boxes"].values())
    assert all(isinstance(position, list) for position in env["goals"].values())
    assert all(isinstance(frame, list) for frame in env["step_frames"])
