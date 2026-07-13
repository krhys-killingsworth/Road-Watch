# AI Usage Log

A running record of AI tools used on this project and how. Update as the project progresses.

| Date | Tool | How it was used | Reviewed / edited by me? |
|------|------|-----------------|--------------------------|
| 2026-07-06 | Claude | Drafted the project proposal (slides, README, repo scaffold, exploration notebook), and confirmed current facts on YOLO versions and the RDD2022 dataset. | Yes — reviewed all outputs, adjusted scope, verified dataset facts. |
| 2026-07-06 | Claude | Wrote the working pipeline code (prepare_data.py, train.py, evaluate.py, detect.py, download_weights.py) and smoke-tested train → evaluate → detect end-to-end on a synthetic dataset. | Yes — code to be re-reviewed as I run it on real RDD2022 data. |
| 2026-07-12 | Claude | Debugged the Week 1–2 Colab run: Roboflow download issues (empty workspace, corrupted zip, API key), path/layout fixes for the Roboflow export, GPU runtime + RAM-crash fixes (batch/workers/cache), class visual-audit workflow, and evaluate/demo cells. | Yes — I ran every cell myself and verified outputs; class names identified by my own visual audit of labeled crops. |
| 2026-07-12 | Gemini (Colab built-in) | Colab's "explain error" assists and one auto-generated path-heuristic cell during dataset debugging. | Yes — its path heuristic was replaced with a simpler verified fix. |
| 2026-07-12 | Claude | Updated repo docs (README, data card, configs, scripts) to reflect the actual dataset used, the 5-class mapping, and real baseline results (mAP@0.5 0.452 / pothole recall 0.646). | Yes — all reported metrics come from my actual training/eval runs, not AI estimates. |
|  |  |  |  |

## Notes / ground rules for myself
- AI is used for scaffolding, drafting, and rubber-ducking — not for fabricating results.
- All metrics reported in the final project will come from **actual training runs**, not AI guesses.
- Any AI-generated code is read and understood before it goes in the repo.
