# Routing sample — random number → 3 branches

The smallest end-to-end demonstration of `workflow.IntRoute` /
`workflow.MultiRoute` and the `Event.Routes` contract. No LLM, no
HITL, no persistence — just routing.

## What it shows

| Concept | Where |
|---|---|
| `FunctionNode` producing a typed value | `roll_die` returns `int` |
| Custom `BaseNode` emitting a routing event | `route_by_value` sets `Event.Routes = []string{fmt.Sprint(value)}` and `Event.Actions.StateDelta["output"] = value` so downstream FunctionNodes get a typed `int` input |
| `MultiRoute[int]` matching a set of ints | three downstream edges, one per range |
| Random behaviour to exercise different paths between runs | `math/rand/v2` in `roll_die` |

`FunctionNode` cannot today emit `Event.Routes` itself (its
wrapper always builds a single output event from the return
value), so the routing node drops down to a custom `BaseNode`.
This is the same limitation called out in the HITL design doc's
"Open question: should `FunctionNode` learn to emit Routes?"
section.

## Run it

```sh
go run ./examples/workflow/route/ console
```

Type any message; the sample ignores it. Each turn rolls a fresh
number. Run a few times in a row to see the route change:

```text
User -> hi
Agent -> rolled 7 — handling MID range (4..7)

User -> hi
Agent -> rolled 2 — handling LOW range (1..3)

User -> hi
Agent -> rolled 9 — handling HIGH range (8..10)
```

## Graph

```
START → roll_die → route_by_value
                      ├─ {1, 2, 3}    → handle_low
                      ├─ {4, 5, 6, 7} → handle_mid
                      └─ {8, 9, 10}   → handle_high
```
