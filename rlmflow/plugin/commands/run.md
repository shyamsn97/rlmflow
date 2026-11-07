---
description: Work on a task as the root agent of an rlmflow run
argument-hint: TASK
---
You are now the root agent of an rlmflow run on this task:

$ARGUMENTS

Work through `rlmflow repl` as the rlmflow context above describes: keep your state in
REPL variables, launch agents for independent pieces, and end your turn to wait for them.
The `orchestrate` skill has the ways to split work across agents and when each pays off.
Call `submit(...)` in the REPL when the task is finished.
