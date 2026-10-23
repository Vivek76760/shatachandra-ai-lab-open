# Shatachandra AI Lab

A free, open-source question-answering agent built around the open Gemma 4 E4B model (4-bit, with Unsloth). It runs on a free Kaggle or Colab GPU, in English, Hindi and Telugu.

Status: early experiment, not a finished product. It is maintained by one person, part-time, so replies may be slow. Please read "Honest results" first.

## What it does
- Short fact answers from Wikipedia and web pages, with a source line.
- Calculation questions: it writes and runs small Python programs. It runs them twice, uses a third run to break ties, and shows a note when the runs disagree.
- Long answers (100 to 600 words) built from Wikipedia sections.
- It often says "I could not find it" when it cannot find an answer.

## Honest results (counted by hand, on tests written by the author)
| Test | Questions | Result |
|---|---|---|
| Code and maths, hard | 40 | 36 right, 3 wrong, 1 no answer. Five of these were fixed by patches after they failed, so expect lower on new questions. |
| Short facts, hard | 50 | 46 to 47 right, 2 wrong, 1 or 2 honest "not found". |
| Long answers, hard | 32 | About 6 with real problems: wrong page, outdated number, junk text. |

A fair planning number is about 85% on hard test questions. That is not 85% on everything. The tests are small, so the margin of error is wide. Larger models do better than this one.

## Known weaknesses
- Some wrong answers come with no warning, for example a wrong reading of the question (a leap day, a loan balance).
- Long answers are copied Wikipedia sentences, so old numbers can appear.
- It is slow: about 15 to 50 seconds per answer, and Wikipedia rate limits cause waits.
- It needs an NVIDIA GPU with about 10 GB of memory in total. It was tested only on Kaggle with two T4 GPUs.

## Do not use it for
Trading, investing, medical, legal or safety decisions.

## How to run
See LOAD_ORDER.md.

## Help wanted: more languages
The agent supports English, Hindi and Telugu. We want more Indian languages (Tamil, Kannada, Malayalam, Bengali, Marathi, Odia and others), and other languages too.

What a new language needs:
- A Wikipedia in that language with enough articles.
- At least 50 hard test questions with answers, written and checked by a native speaker (see the files in tests/).
- Code. Today Hindi and Telugu each have their own copy of the code (request detection, page lookup, digit table, filler words). A new language currently means copying that code, which is not elegant. Help making this cleaner is very welcome.

Please have a native speaker check every answer. Answer keys written by someone who does not read the language well contain mistakes.

## Other ways to help
- Run the tests in tests/ and send us your results.
- Report wrong answers: open an issue with the question, the answer you got, the right answer, and a source.
- Work on the known weaknesses above.

Rules: never commit secrets or kb.json, and report test results honestly, including failures.

## Support the project
The code is free under Apache 2.0, and a star on the repo is the best free support. If this project helped you and you want to support the work with money, a donation link will be added here soon. Donations are welcome but never required.

## Co-founders and partners
If you want to help turn this into a real product or a company, for example for schools, coaching centres or tutoring in Indian languages, we would like to hear from you. Open an issue titled "Co-founder:" with a short note about who you are and what you can build, teach or sell.

## Licence and credits
- Code: Apache License 2.0 (see LICENSE). Anyone, including companies, may use and sell products built on this code under that licence.
- Model: Google Gemma 4 E4B, Apache 2.0. It is not included; the code downloads it from Hugging Face.
- Answers copy text from Wikipedia under CC BY-SA. Keep the source line the agent prints.
- Web pages keep their own copyright.
- The agent saves answers to a local kb.json. Do not publish that file; it is unreviewed.
