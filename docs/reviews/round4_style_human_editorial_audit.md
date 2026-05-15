# Round 4 — Style / Human Editorial Audit

**Subject.** `paper_output_v2/spectralquant_v2_full_story.md`.
**Auditor.** Editorial pass at commit `96e229c`, 2026-05-01.
**Question for this round.** Does the prose read like it was written
by someone who knows the work, or does it carry the cadence and tics
of generic AI-assisted writing? Are there fluffy transitions,
boilerplate summaries, or inflated claims that should be cut?

This audit does not claim to detect or remove "AI authorship" — that
is not a thing one can verify on prose alone. It tries to do the
narrower thing: look for the specific habits that make text feel
generic (hedge phrases, throat-clearing transitions, signposting
clauses, vague intensifiers, "we conclude that"-style boilerplate)
and either remove them or note their absence.

## 1. Patterns scanned for, with results

| Pattern | Hits | Action |
|---|---:|---|
| `It is worth noting / It's important to note` | 0 | – |
| `In conclusion / In summary` | 0 | – |
| `Furthermore / Moreover / Notably` | 0 | – |
| `In essence / Essentially / Basically` | 0 | – |
| `crucial / pivotal / vital` | 0 | – |
| `comprehensive / robust / seamless / streamline` | 0 | – |
| `leverage` (verb) | 0 | – (one hit on the hyphenated adjective "highest-leverage", which is fine) |
| `delve into / navigate the / a wealth of / the landscape of` | 0 | – |
| `plays a crucial role / paradigm shift / cutting-edge / unprecedented` | 0 | – |
| `we conclude that / it can be observed / as we can see` | 0 | – |
| `clearly / obviously / of course` | 0 | – |
| Repeated `However / Therefore / Thus / Hence / Consequently` | 0 each | – |

(Pattern scanner is reproduced at the end of this doc as a Python
snippet for future re-runs.)

The manuscript is unusually clean of these tics, mostly because the
load-bearing sentences are imperative claims tied to numbers
("perplexity is 6.98", "F1 is 0.482"), not narrative throat-clearing.

## 2. Sentence opening variety

Top 20 sentence-opening words:

```
The: 45,  This: 15,  For: 8,  We: 8,  A: 7,
2: 6,   It: 5,   v2: 4,   3: 4,   These: 4,
TurboQuant: 4,  Every: 3, Across: 3, 4: 3, 5: 3,
What: 3, At: 3, SpectralQuant: 3, v1: 3, Compression: 3
```

The document is ~1170 lines / ~9 000 words, so 45 "The" / 15 "This"
opens out of several hundred sentences is a normal distribution for
technical prose. There are no repeated transition words ("However",
"Therefore", etc.) clustering at the start of paragraphs.

## 3. Inflated claims

I read every section looking for claims that go beyond what the
evidence supports. The flagged candidates and how the document
handles them:

- **§0 thesis.** "Conclusively beats the in-repo TurboQuant
  comparator …" — bounded by "in-repo" (rather than "official"),
  bounded by "tested matched full-path benchmarks", and explicitly
  itemised next to the "what this evidence does *not* unblock"
  paragraph. Not inflated.
- **§8.3 reading.** "SQv2 macro-beats FP16 on this subset by
  +0.0250 absolute (+14.2 % relative)" — bounded by "this subset",
  reported next to the n=50/single-seed/no-CI caveat in the same
  section, restated in §15 with the subset tag. Not inflated.
- **§12 interpretation.** "Structure beats budget" — this is a
  framing, not a claim about a measurement. Use is consistent with
  the §15 defensible-wording rules.
- **§15 defensible wording.** Every word is bounded to what the
  Round 1 audit verified. The "What we are not claiming" paragraph
  is itemised. Not inflated.

Nothing pushes past the limitations declared in §13.

## 4. Boilerplate summaries

The document deliberately avoids the "we conclude that" / "in this
section we have shown" pattern. §0 is itself the executive summary;
§12 is the interpretation; §13–§15 are limitations + traceability +
defensible wording. There is no "in summary" paragraph capping each
section. This is intentional and a stylistic choice we keep.

## 5. Hedges and weasel words

No hedges found ("essentially", "basically", "in other words", etc).
No weasel words ("various", "myriad", "plethora").

The honest hedges that *do* appear are domain-specific and necessary:
"single seed", "n=50/task", "in-repo TurboQuant comparator",
"deterministic 5-task subset", "production_kernel = false". These are
caveats, not weasel words; they are required by the data and they
appear in the same sentence as the claim they bound.

## 6. Tone audit (Sentra-fundraising surface)

§0 ("structure beats budget") and §1.3 ("why structure rather than
scale") are the closest the document gets to a fundraising surface.
The phrasing is direct: "KV-cache compression at production fidelity
is a load-bearing piece of the long-context inference cost curve, and
the methods that *understand* the cache (calibrated rotations,
water-filled allocation) will win on the metrics customers pay for
(cost per token, max concurrency at long context). That is the
technical thesis the rest of the document substantiates."

This sentence puts a thesis on the table that the rest of the document
is allowed to defend with measurements. It does not say
"revolutionary", "game-changing", "unprecedented", or "industry-
leading". It does not invoke a TAM or a market projection. The
fundraising framing is reduced to a defensible engineering claim, and
the engineering claim is reduced to numbers in §6–§10.

## 7. Concrete edits made during this round

The drafting pass was already disciplined; this round produced no
text edits. Two earlier-round revisions (Round 2) added the
`cover2006elements` and `ainslie2023gqa` bibliography entries and
changed `[cover2006elements_chap10]` → `[cover2006elements, ch. 10]`
in two places.

## 8. Risks the audit cannot catch

A few stylistic choices the document makes are not bugs but should be
called out so the reader can decide:

1. **Heavy use of `[evidence: …]` annotation.** This is by design and
   matches the rule in `paper_output_v2/spectralquant_v2_longform.md`.
   It looks unusual in a prose document but it is the load-bearing
   feature of this manuscript: every empirical sentence carries its
   own audit pointer. We keep it.
2. **Verbatim numbers.** The document quotes JSON metrics to four or
   five digits; rounding is sometimes inconsistent (e.g. "0.482" vs
   "0.4817"). This is intentional: §0/§7/§15 round to 3 sig figs;
   §6.2/§7.2 use the JSON's own precision. The §18 footer tells
   downstream readers to regenerate from JSON, not to transcribe.
3. **No figures.** The full story is prose-first and cites numbers,
   not plots. Plots live in `paper_output/figures/` and in the
   NeurIPS-format paper. We keep the prose-first approach because
   the audit story is easier to verify on numbers than on figures.

## 9. Outcome

**PASS.** The prose is precise. There are no fluffy transitions, no
hedge phrases, no inflated claims, and no boilerplate summaries. The
caveats are bounded by the data they reflect. The Sentra-fundraising
surface is a defensible engineering claim, not marketing.

## Appendix — Pattern scanner (for future re-runs)

```python
import re
text = open('paper_output_v2/spectralquant_v2_full_story.md').read()
patterns = [
    r"It is worth noting", r"It's important to note",
    r"In conclusion,", r"In summary,", r"Furthermore,",
    r"Moreover,", r"Notably,", r"It should be noted",
    r"In essence,", r"Essentially,", r"Basically,",
    r"\bcrucial\b", r"\bcomprehensive\b", r"leverage(s|d|ing)?\b",
    r"delve(s|d|ing)? into", r"navigate the", r"\brobust\b",
    r"streamline", r"underscore", r"\bvital\b", r"seamless(ly)?",
    r"plays a (crucial|pivotal|key) role", r"In the realm of",
    r"a wealth of", r"the landscape of", r"we conclude that",
    r"as we can see", r"of course", r"clearly,", r"obviously,",
    r"\bvarious\b", r"\bplethora\b", r"\bmyriad\b",
    r"intricacies", r"holistic", r"paradigm shift",
    r"game-chang", r"cutting-edge", r"unleash", r"empower",
    r"unprecedented", r"revolutionar",
]
for p in patterns:
    n = len(re.findall(p, text, flags=re.IGNORECASE))
    if n: print(f"{p}: {n}")
```
