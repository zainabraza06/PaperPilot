"""Prompts for paper summarization.

Kept in one module, versioned, and separate from any provider. Two reasons
beyond tidiness: the cache key includes ``PROMPT_VERSION``, so changing an
instruction here can never serve text produced under the old one; and the
grounding check and the prompt have to agree about what is being asked
for, which is easier to keep true when both are visible in one place.

The instructions are written against the failure modes the checker looks
for. That is deliberate — the cheapest way to pass a grounding check is to
ask for grounded output in the first place, and the check is the backstop
rather than the mechanism.
"""

from __future__ import annotations

from app.models.summary import GroundingIssue

#: Bump on any change below. Cached summaries from older versions are
#: ignored rather than served, so a prompt fix takes effect immediately.
#:
#: v2 forbids markdown emphasis. The model was returning "replaces
#: **CRISPR-Cas9** with the smaller **Cas12a**", which a web UI renders as
#: literal asterisks.
#:
#: A v3 was written and thrown away, which is worth recording so it is not
#: tried again. The quality harness showed these summaries carrying less
#: of the abstract than the extractive fallback does (coverage 0.658
#: against 0.730 on 94 live papers), so v3 asked for the question, the
#: method *and* the finding, and added a rule about not stopping after the
#: background. Re-measured on 100 fresh papers it moved coverage by
#: nothing at all -- 0.658 to 0.658 -- while compression rose from 0.428
#: to 0.464 and redundancy from 0.556 to 0.597. The summaries got roughly
#: eight percent longer and carried exactly as much. Lead bias got
#: slightly worse too (0.601 to 0.569), so the added rule did not even buy
#: the thing it named.
#:
#: The coverage gap is therefore not a prompt problem, and asking harder
#: is not the lever. It is what a 2-3 sentence budget costs: three
#: sentences of compressed prose hold less of an abstract than three
#: sentences lifted from it, and that is the trade the feature makes.
PROMPT_VERSION = 2

SYSTEM_PROMPT = """You summarize scientific paper abstracts for researchers.

Rules, all of which are checked automatically:
- Write 2 to 3 sentences. No more.
- Use ONLY information stated in the abstract. Add nothing from your own \
knowledge of the field, the authors, or the topic.
- Do not state any number, percentage, gene name, method name or acronym \
that does not appear verbatim in the abstract.
- Do not reverse or soften a direction of effect. If the abstract says \
something increased, do not write that it decreased or changed.
- Do not claim novelty, superiority or certainty ("the first", "proves", \
"best") unless the abstract makes that claim itself.
- If the abstract is vague, be vague. Do not fill gaps.

Write plain prose. No preamble, no bullet points, no headings, no citation \
markers, and no markdown formatting of any kind - no **bold**, no *italics*, \
no `code`. Return only the summary text."""


def build_user_prompt(title: str, abstract: str) -> str:
    """The first-attempt prompt for one paper."""
    return (
        f"Title: {title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Summarize this abstract in 2-3 sentences for a researcher deciding "
        "whether to read the full paper. State what was done and what was found."
    )


def build_retry_prompt(
    title: str, abstract: str, previous: str, issues: list[GroundingIssue]
) -> str:
    """The second-attempt prompt, with the specific failures fed back.

    Showing the model its own rejected output and the exact reasons is far
    more effective than re-running the original prompt and hoping for a
    different sample — and unlike raising the temperature, it makes the
    retry a correction rather than a re-roll.
    """
    problems = "\n".join(f"- {issue.detail}" for issue in issues)
    return (
        f"Title: {title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"A previous attempt at this summary was rejected:\n\n"
        f'"{previous}"\n\n'
        f"It was rejected for these specific reasons:\n{problems}\n\n"
        "Write a corrected 2-3 sentence summary that avoids every problem "
        "listed above. Use only what the abstract states."
    )
