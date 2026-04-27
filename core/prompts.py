SYNTHESIS_TEMPLATE = """Task:
  Fill in the typed slots of the given skeleton with API calls drawn from the candidate pool, producing one complete, executable program.

Inputs:
  - Target library: {target_lib}
  - Skeleton template:
{skeleton}
  - Candidate API pool:
{api_pool}

Requirements:
  1) Preserve the skeleton. Do not modify the Model scaffold, the state-installing constructs, or the driver logic.
  2) Fill the body slot. Compose APIs drawn from the candidate pool into a dependency chain within the model body, using each API at most once.
  3) Fill the input slot. Provide a randomly initialized tensor matching the body's expected input shape and dtype.
  4) The resulting program must be syntactically valid and runnable as a standalone file.

Output:
  - Return ONLY the complete program code.
"""


REPAIR_TEMPLATE = """The program failed with the following error: {error_brief}

Repair instructions:
  1. Fix the error with minimal edits.
  2. Do not alter the skeleton scaffold: Model class structure, state-installing constructs, and driver logic must be preserved.
  3. Changes are allowed only within the body slot and the input slot (e.g., adjusting API arguments, input shapes or dtypes, or replacing a problematic API call with another from the original candidate pool). The skeleton scaffold must remain unchanged.

Original program:
{original_code}

Return the FULL corrected program only.
"""


def _format_pool(api_list: list[str]) -> str:
    return "\n".join(f"    - {a}" for a in api_list)


def _format_skeleton(skeleton: str) -> str:
    return "\n".join("    " + line for line in skeleton.splitlines())


def build_synthesis_prompt(
    api_list: list[str],
    skeleton: str,
    target_lib: str = "PyTorch",
) -> str:
    return SYNTHESIS_TEMPLATE.format(
        target_lib=target_lib,
        skeleton=_format_skeleton(skeleton),
        api_pool=_format_pool(api_list),
    )


def build_repair_prompt(original_code: str, error_brief: str) -> str:
    return REPAIR_TEMPLATE.format(
        error_brief=error_brief[:1500],
        original_code=original_code,
    )


def build_tf_synthesis_prompt(api_list: list[str], skeleton: str) -> str:
    return build_synthesis_prompt(api_list, skeleton, target_lib="TensorFlow 2.x")


def build_tf_repair_prompt(original_code: str, error_brief: str) -> str:
    return build_repair_prompt(original_code, error_brief)
