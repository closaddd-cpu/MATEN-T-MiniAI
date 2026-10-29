# Dataset Card — BY MATEN/T

94 independent UTF-8 JSONL shards; 100,000 records/shard; 9,400,000 total records. Original procedural generator `build_datasets.py` and manually drafted starter text `catalog.py`, `conversation_human.py`, `facts.py`.

- **Language scope:** 50 programming names and 20 written natural-language shards.
- **Subject scope:** 24 named science/defensive-security/conversation subjects.
- **Text provenance:** no downloaded third-party corpora or scraped conversations; automatically generated examples from original templates with elementary commonly known facts. Does **not** imply exclusive copyright over facts, nor rights in external libraries or user data.
- **Quality:** many rows vary numeric operands within small repeated templates. Native-speaker translations have **not** been independently reviewed, and most programming languages have **not** been compiler-tested. Some syntax may be version-specific, incomplete, or incorrect. Do not claim domain mastery, sophisticated conversations or production-grade correctness on these numbers.
- **Data split:** deterministic index modulo 20: train 90%, validation 5%, test 5%; related template families can appear in all splits, creating data leakage and inflated validation performance. External family-held-out testing needed.
- **Security:** only defensive cybersecurity guidance; not an intrusion toolkit.
- **Privacy:** no personal conversation data harvested or bundled.
- **License:** creator-owned project portions under custom M/ License v1.0 (not OSI approved); other dependencies/licenses unaffected.
- **Record fields:** `id, domain, topic, instruction, response, created_by, license, provenance, review_status`.

**Recommended next stage:** recruit native-speaking contributors for human-reviewed dialogs, gather consented/open-license source materials with attribution, test code across actual interpreters/compilers, de-duplicate templates, and construct independently authored held-out evaluations.
