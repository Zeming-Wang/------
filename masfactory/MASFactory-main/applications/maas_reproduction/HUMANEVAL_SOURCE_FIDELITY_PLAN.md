# HumanEval Source-Fidelity Implementation Plan

## Objective

Make the MASFactory reproduction execute HumanEval with the source MaAS semantics while preserving MASFactory's graph, node, immutable-state, dependency-injection, and hidden-test-isolation conventions.

This work does not change `sample` semantics, ordinary-run operator embedding construction, or sequential dataset execution.

## Compatibility baseline

The default HumanEval compatibility layer implements source behavior. It does not add per-operator retries, JSON or bare-letter ScEnsemble fallbacks, stricter code-only instructions, a stricter sandbox, or distinct GenerateCoT instructions.

Two intentional deviations are documented:

1. Candidate execution remains isolated in a subprocess with a process-wide 15-second timeout.
2. Whole-architecture execution is attempted at most three times instead of the source benchmark's twenty attempts.

## Required execution flow

1. The Controller produces a `RoutePlan` over the source-ordered HumanEval catalog:
   `Generate`, `GenerateCoT`, `MultiGenerateCoT`, `ScEnsemble`, `Test`, `SelfRefine`, `EarlyStop`.
2. Route operators execute in order. A selected `Test` invokes the source-compatible public-test and repair protocol.
3. After route execution, HumanEval always invokes the same Test protocol again.
4. The Test protocol performs at most three reflection repairs and a final fourth public-test execution.
5. If the fixed final Test still fails, `CustomCodeGenerate` is called once with `IMPROVE_CODE_PROMPT + problem`.
6. The fallback candidate is not publicly retested; it is sent directly to the hidden HumanEval evaluator.

## Source-compatible code fill

Provide one deep module with this interface:

```python
CodeFillAdapter.fill(*, prompt: str, sanitize_entrypoint: str | None) -> str
```

It invokes the model without new code-only instructions and then applies the source `sanitize` implementation.

| Operator | Prompt | sanitize entrypoint |
| --- | --- | --- |
| Generate | `problem` | `entry_point` |
| GenerateCoT | `problem` | `entry_point` |
| MultiGenerateCoT | `problem`, independently three times | `entry_point` |
| CustomCodeGenerate | `IMPROVE_CODE_PROMPT + problem` | `entry_point` |
| Test reflection | source reflection prompt | `None` |
| SelfRefine | source SelfRefine prompt | `None` |

Generate, GenerateCoT, and each MultiGenerateCoT call intentionally use the same empty instruction in compatibility mode.

## Source-compatible execution

Port the source MaAS sanitizer rather than extending the reproduction's restrictive AST implementation. Hidden evaluation retains standard builtins and imports and prepopulates `math`, `hashlib`, `re`, `List`, `Dict`, `Tuple`, `Optional`, and `Any`, together with the three source special helpers.

The process-wide timeout is retained and recorded as an intentional deviation.

## Public-test repository

Public tests are loaded by entry point from the source public-test file and source hardcoded cases. Hidden `EvaluationContext.test` and `canonical_solution` never cross into architecture execution.

```python
HumanEvalPublicTestRepository.get(entry_point: str) -> tuple[str, ...]
```

Source quirks, including empty hardcoded public-test cases, are preserved.

## HumanEvalTestGraph

`HumanEvalTestGraph` is the single module used both by the route `Test` operator and fixed completion.

```text
execute public tests
  -> pass: return current solution
  -> fail and repairs < 3: reflect, sanitize without entrypoint, repeat
  -> after third repair: execute a fourth and final public test
```

A completed protocol returns `OperatorResult(status="completed")` whether public tests pass or fail. Public-test failure is business data; only model, parser, repository, or executor infrastructure failures use `status="failed"`.

## HumanEval completion

`ArchitectureExecGraph` selects `HumanEvalCompletionGraph` for HumanEval. The graph invokes `HumanEvalTestGraph` unconditionally. A passing result becomes the prediction. A failing result invokes exactly one source-compatible fallback generation and returns it without another public test.

## ScEnsemble

Compatibility mode accepts only the source XML field:

```xml
<solution_letter>A</solution_letter>
```

The value must be one letter and address an existing candidate. JSON, bare-letter, and first-uppercase-character parsing are not part of compatibility mode.

## Whole-architecture attempts

Ordinary operators remain single-attempt. Retry is placed at the source benchmark seam and reruns Controller sampling, route planning, route execution, fixed completion, and all associated costs. `graph_max_attempts` is three total attempts. Normal incorrect code or score zero does not trigger another attempt; only retryable infrastructure/model/parser failures do.

## HumanEval source bundle v2

The source exporter and reproduction loader add the HumanEval catalog. GSM8K/MATH continue accepting v1; HumanEval requires v2.

HumanEval v2 contains and validates:

- exact Controller state dict and controller specification;
- exact ordered operator catalog;
- source description/interface operator embeddings;
- embedding model and query embeddings;
- source controller and operator JSON provenance;
- workflow contract identifier;
- per-layer complete probability and log-probability fixture vectors;
- previous-operator indices, selected indices, and aggregate log probability.

Fixtures are captured from the real source Controller with hooks, not from a reimplemented reference Controller. Reproduction validation compares full vectors with explicit tolerances before test execution.

## Verification

Tests cover code-fill entrypoint differences, identical Generate/GenerateCoT behavior, three independent MultiGenerateCoT calls, strict XML ScEnsemble, source imports/globals, public-test repair counts, route Test registration, unconditional fixed Test, one untested fallback, hidden-test isolation, three whole-graph attempts, HumanEval v2 validation, per-layer Controller distribution parity, and fixed-response end-to-end prediction/score parity.

## Implementation order

1. Source sanitizer, execution adapter, public-test repository, prompts, code-fill, and strict XML parsing.
2. `HumanEvalTestGraph` and `Test` registration.
3. `HumanEvalCompletionGraph`, fixed Test, and fallback wiring.
4. Whole-architecture three-attempt policy.
5. HumanEval source bundle v2 and Controller distribution fixtures.
6. Unit, graph, integration, and fixed-response differential tests.
