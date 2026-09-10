# MaAS Reproduction Context

This context defines the shared language for the native MASFactory reproduction
of MaAS.  It separates policy decisions, execution state, public results, and
training eligibility.

## Policy and execution

**RoutePlan**:
The flattened operator route selected by the policy controller, together with
the aggregate policy log-probability for that route.
_Avoid_: execution state, operator trace

**RouteItem**:
A single ordered operator or the deterministic EarlyStop control marker in a
RoutePlan.
_Avoid_: step, action result

**DispatchState**:
The minimal state needed to continue a selected route and choose its final
prediction.
_Avoid_: global workflow state, execution trace

**OperatorResult**:
The structured business result of one operator invocation, including a
recoverable failure when one can be represented.
_Avoid_: exception, dispatch state

## Results and training

**ArchitectureResult**:
The final execution result before dataset-specific scoring, including result
validity and cost reliability.
_Avoid_: evaluation result, sample result

**EvaluationResult**:
An ArchitectureResult enriched with the dataset score and evaluator
reliability.
_Avoid_: reward, training signal

**TrainingSignal**:
The single decision that says whether an evaluated sample is eligible for a
policy update and, if so, its utility.
_Avoid_: advantage, failure penalty

**SampleResult**:
The detached public record retained for a completed sample; it never carries a
live policy Tensor.
_Avoid_: live training result, architecture result

**FailureSource**:
The location category of a failure used for control, logging, and diagnosis;
it never changes the utility formula.
_Avoid_: failure reward, penalty source
