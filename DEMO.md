# GlassBox demo script

How to show that GlassBox works in about 15 minutes, without running anything long.

**The idea:** the *system* answers a question in about 3 seconds, so everything the professor sees in the demo runs live. The *experiments* (390 questions × 3 configurations, the 2.5-hour threshold sweep) were run beforehand. Their results are shown, and a one-minute check proves those stored results come from the code as it is.

---

## Before the meeting (10 minutes)

1. Plug the laptop in. Close other heavy apps: the 7B model needs about 5 GB of memory and the GPU.
2. Open a terminal in the repo:
   ```bash
   cd GlassBox && source .venv/bin/activate
   ```
3. Start the dashboard. It opens at http://localhost:8501 and listens only on this laptop.
   ```bash
   streamlit run glassbox/src/dashboard/app.py
   ```
4. On the **Demo** page, press **Load models** (about 20 s). Then press **Run** on step 1 once to warm everything up.
5. In a second terminal tab, have `python -m glassbox.verify` ready to run (don't run it yet).
6. Have the backup screen recording ready, in case anything fails on the day.

---

## The demo (about 15 minutes)

### Part A: the problem and the idea (1 minute, no laptop)

> LLM question answering over medical records has two problems: the model sees patients' identities, and you can't tell whether its answers are true. GlassBox is a pipeline where privacy and verifiability are enforced and *measured*. It redacts before anything sees the data, retrieves only what's needed, makes the model cite every claim, checks every claim with a second, independent model, and logs every step so it can be audited.

All data is synthetic: 2,500 generated clinical notes. There are no real patients.

### Part B: live questions (Demo page, 8 minutes)

Each step is one button and takes about 3 s. "Show talking points" displays what to point out under each result. For any step, turn on **Full audit trace** to walk through every stage: redaction, filters, retrieved chunks with scores, the answer, and per-claim verification.

| Step | What it shows | Key line |
|---|---|---|
| 1 · Clinical fact | Filters from the question, one cited claim, verified SUPPORTED; the chunks contain only `[NAME]`-style placeholders | "The model answered without ever seeing who the patient is, and a second model checked the answer." |
| 2 · Several patients | Two-hop retrieval (symptom → that patient's medication section); one verified claim per patient | "Each claim cites its own patient's note." |
| 3 · Named person | **Left:** GlassBox redacts the name and refuses before the model runs. **Right:** the raw baseline gives that person's medication. | "Same pipeline, privacy off: it leaks. With privacy on, it answered 0 of 84 such test questions." |
| 4 · No matching record | Refuses: no asthma note exists on that date | "It says 'no records match' instead of answering from the wrong notes." |
| 5 · Try to fool the verifier | Edit step 1's claim (change the dose, swap the drug, add a fact) and press Verify: CONTRADICTED / UNSUPPORTED | "Let *you* try to sneak a false claim past it." Hand the keyboard over. |
| 6 · Your own question | The professor types anything | Suggest: a diagnosis + month, a symptom across patients, or a name |

Good questions for step 6 (all work on this data):
- *Which medications were prescribed to COPD patients who presented with sputum production in July 2025?*
- *What symptom or complaint was documented for the COPD patient whose note is dated 2024-11-19?*
- Any question containing a person's name: it is refused.

### Part C: it works in general, not just on these questions (4 minutes)

1. **Stored traces page.** All 390 evaluation questions, already run. Filter by verdict or question type, click any row and audit it.
   - Answerable test questions: **0.858** grounded accuracy (answer correct *and* cited from the right patient's note), 95% CI [0.815, 0.899].
   - Identity questions answered: **0 of 84** (raw baseline: answered 42, and gave the person's true value for 16).
   - Answers the verifier marks VERIFIED are right 0.906 of the time, vs 0.492 for NOT_VERIFIED, so the verdict is a useful trust signal.
2. **Experiments page** (figures with confidence intervals): redaction quality, leakage, privacy vs utility across redaction thresholds (the 2.5-hour sweep, shown as its result), verifier detection rates, robustness to unseen wording.
3. **Live proof the stored numbers are real.** In the terminal:
   ```bash
   python -m glassbox.verify            # ~1 minute
   ```
   It re-runs 10 stored evaluation questions through the whole pipeline and prints `10/10 identical to the stored evaluation run`: same chunks, same answer word for word, same citations, same score. Decoding is deterministic, so any edited or cherry-picked result would show up as a difference. Add `--tests` to also run the 165 automated tests (+30 s).

### Part D: honest limitations (1 minute)

Saying these yourself is stronger than being asked:
- **Synthetic data**, generated by templates. On held-out paraphrased questions, accuracy drops from 0.858 to 0.502; on unseen note formats, redaction drops from 0.999 to 0.819. Both were measured with a held-out set frozen before any fix.
- **Verification checks evidence, not identity.** Without the guard, the verifier still marks most misattributed claims SUPPORTED, so the guard (before the model) is what protects identities.
- **Redaction, not formal privacy guarantees** (no differential privacy). It runs on a local 7B model.

---

## Likely questions

| Question | Answer |
|---|---|
| Why refuse instead of answering "the patient takes X"? | Notes are de-identified, so the system can't know which note belongs to the person named. Without the guard, the model answered 84 of 84 such questions, *all from other people's notes*. |
| What if the verifier is wrong? | It catches 99.4% of corrupted claims on a test set of 1,289, and wrongly flags 1.2% of clean genuine claims. When it is wrong it is confidently wrong (a known calibration limitation, reported). |
| How do I know these numbers aren't cherry-picked? | Design choices were made on a dev split and reported on a separate test split with confidence intervals; robustness sets were committed to git *before* fixes (commits `2f0bb6b`, `a0047eb`); `glassbox.verify` reproduces stored results live. |
| Why not just put "don't reveal identities" in the prompt? | Prompts can be ignored or argued around. The guard is code that runs before the model, and its decision is logged. |
| Does redaction hurt accuracy? | No measurable cost: retrieval evidence 0.949 redacted vs 0.907 raw; grounded accuracy 0.858 vs 0.833. |
| How fast is it? | About 3.2 s per question; about 98% is the 7B model. Redaction (4 ms), retrieval (6 ms) and verification (37 ms) together add under 2%. |
| Could it use real records? | The pipeline doesn't depend on the data source, but real records would need ethics approval and stronger redaction validation. Out of scope by design. |

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Dashboard won't start | `pkill -f streamlit`, then start it again |
| First question is slow | Models weren't loaded: press **Load models** and wait ~20 s |
| Laptop is slow or out of memory | Close other apps; or switch to the **Stored traces** page, which needs no models |
| Anything else | Play the backup recording; walk through stored traces |
