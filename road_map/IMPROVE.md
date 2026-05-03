Yes. Below is a **complete workflow for a local subtitle translation pipeline**, designed around these 3 goals:

1. **Reduce hallucinations**: do not invent content
2. **Reduce omissions**: do not skip sentences or phrases
3. **Preserve subtitle timeline alignment**: output should still map cleanly to the original subtitle segments

I’ll describe it in a **practical engineering workflow** so you can implement it directly.

---

# 1. Overall Workflow

```text
Audio / Video
  ↓
ASR transcription (with timestamps preserved)
  ↓
Segment normalization (clean text without changing timeline)
  ↓
Pre-translation checks (length, empty lines, abnormal characters)
  ↓
Segment-by-segment translation (with strict prompt constraints)
  ↓
Post-translation validation (hallucination / omission checks)
  ↓
Automatic retry when needed
  ↓
Subtitle compression / line breaking optimization
  ↓
Export SRT / VTT / JSON
```

---

# 2. What Each Step Should Do

## 1) ASR Transcription: Preserve the Smallest Useful Time Granularity

If the input is a WAV file, the first step should output something like this:

```json
[
  {
    "id": 1,
    "start": 0.00,
    "end": 2.14,
    "text": "This is a record loss for our military forces."
  },
  {
    "id": 2,
    "start": 2.15,
    "end": 4.90,
    "text": "We have never seen anything like this before."
  }
]
```

### Key principles

- Every segment must have `start` and `end`
- During translation, **only change `text`, never the timestamps**
- Do not merge too many segments into large blocks too early, or omissions and misalignment become more likely

### Recommendation

- If `faster-whisper` produces segments that are too fragmented, apply light merging
- But ideally each segment should stay within:
  - **1.5s to 6s**
  - with moderate text length

---

## 2) Segment Normalization: Clean Only, Do Not Rewrite

The goal here is to make inputs more stable **without changing meaning**.

### Safe operations

- Trim leading/trailing whitespace
- Normalize repeated spaces
- Remove obvious ASR noise such as standalone `...` or redundant spaces
- Preserve original casing, proper nouns, and numbers

### Avoid

- Do not use an LLM to rewrite the source text first
- Do not polish or paraphrase in advance
- Do not over-edit punctuation

### Best practice

Keep both fields:

```json
{
  "source_text_raw": "...",
  "source_text_norm": "..."
}
```

That way, if translation behaves badly later, you can still fall back to the raw version.

---

## 3) Pre-Translation Checks: Block High-Risk Segments Early

Before sending a segment to the translator, run simple rule-based checks.

### Checks

- Empty text
- Punctuation-only text
- Single-word noise
- Extremely long sentences
- Abnormal repeated characters
- Very short duration but too much text

Example:

```python
def should_skip(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if all(ch in ".,!?-—:;()[]{}\"' " for ch in stripped):
        return True
    return False
```

### Why this matters

A lot of hallucinations are caused not by weak translation models, but by unstable or low-quality input.

---

# 3. Translation Layer: Core Design

## 4) Translate by Segment, Not the Whole File at Once

The most reliable approach is not to feed the entire SRT file to the model.

### Recommended strategy

- **Single-segment translation**
- Or **small-batch translation (3 to 8 segments)**
- But always require the model to return output by ID

Example input:

```json
[
  {"id": 15, "text": "This is a record loss for our military forces."},
  {"id": 16, "text": "We have never seen anything like this before."}
]
```

Expected output:

```json
[
  {"id": 15, "translation": "This is a record loss for our military forces."},
  {"id": 16, "translation": "We have never seen anything like this before."}
]
```

### Why not use plain text output

Plain text output is much more likely to:

- skip lines
- merge lines
- add explanations
- reorder content

---

## 5) Translation Prompt: Strong Constraints + ID-Based Output

Suggested system prompt:

```python
system_prompt = f"""
You are a professional subtitle translator.

Translate each subtitle item into {target_lang}.

Rules:
- Translate faithfully and completely.
- Do not omit, summarize, soften, or add content.
- Preserve the tone and meaning of the original.
- Keep each item's id unchanged.
- Return exactly one translation for each input item.
- Do not merge or split items.
- Output valid JSON only.
"""
```

Suggested user prompt:

```python
user_prompt = f"""
Translate the following subtitle items into {target_lang}.

Return JSON in this exact format:
[
  {{"id": 1, "translation": "..."}}
]

Input:
{json.dumps(batch, ensure_ascii=False)}
"""
```

---

# 4. Anti-Hallucination Mechanisms

This is the most important part.

## 6) Validate Structure After Translation

Do not use model output directly. Validate it first.

### Required checks

- Is it valid JSON
- Does the number of IDs match
- Do the IDs map one-to-one
- Is any `translation` empty
- Are there extra fields
- Is there extra explanation text outside the JSON

Example:

```python
def validate_structure(src_items, out_items):
    src_ids = [x["id"] for x in src_items]
    out_ids = [x["id"] for x in out_items]
    return src_ids == out_ids
```

### If validation fails

Retry immediately. Do not continue to the next stage.

---

## 7) Check Content Fidelity

Typical hallucination patterns:

- adding information not present in the source
- turning a short sentence into a long explanation
- converting uncertainty into certainty
- mistranslating named entities

### Useful checks

#### A. Length ratio check

The translated text should not be disproportionately longer or shorter than the source.

For English → Chinese, for example:

- Chinese is often slightly shorter than English or similar in information density
- If an 8-word source becomes 40 Chinese characters, that is suspicious

Example rule:

```python
def suspicious_length(src: str, tgt: str) -> bool:
    src_len = len(src.strip())
    tgt_len = len(tgt.strip())
    if src_len == 0:
        return True
    ratio = tgt_len / src_len
    return ratio > 3.0 or ratio < 0.2
```

#### B. Key entity check

Extract and verify whether the translation preserves:

- numbers
- years
- percentages
- person names
- place names
- capitalized proper nouns

Example source:

```text
NATO lost 3 vehicles in 2024.
```

Check whether these survive:

- NATO
- 3
- 2024

If all of them disappear, the translation is suspicious.

#### C. Sensitive polarity checks

Words like:

- yes / no
- never / always
- win / lose
- support / oppose

If these are translated with reversed meaning, the impact is serious.
You can maintain a small rule list for spot checks.

---

## 8) Back-Translation Validation: Very Effective Against Hallucination

You can add a lightweight second pass:

```text
Original English → translated Chinese → back-translated English
```

Then compare:

- whether key entities remain
- whether the meaning has drifted significantly

### Best used for

- high-risk segments
- long sentences
- sentences with numbers or named entities
- cases where previous validation already failed

### Not recommended for all segments

It will slow the pipeline down too much.

---

# 5. Anti-Omission Mechanisms

## 9) Strong Count Validation by Item

This is one of the simplest and most effective checks:

- If input has 5 items
- output must also have 5 items
- and the IDs must match exactly

If even one is missing, treat it as failure and retry.

This is much more reliable than judging by appearance.

---

## 10) Check for Untranslated Source Text

Common omission patterns:

- direct copy of the source
- mixed-language output
- partially untranslated content

You can add a simple rule:

```python
def looks_untranslated(src: str, tgt: str) -> bool:
    src_clean = src.strip().lower()
    tgt_clean = tgt.strip().lower()
    return src_clean == tgt_clean
```

Also combine it with:

- English-character ratio checks
- flags for long untranslated English phrases inside Chinese output

---

## 11) Retry Empty or Overly Short Output

Example source:

```text
This is a record loss for our military forces.
```

If the output is:

```text
Loss.
```

it is not empty, but clearly incomplete.

### Rule idea

Retry if:

- source exceeds N words
- translation is fewer than M characters
- or key information is missing

---

# 6. Timeline Alignment

## 12) Never Change Timestamps During Translation

This is the single most important rule.

Each subtitle item should keep this structure:

```json
{
  "id": 1,
  "start": 0.00,
  "end": 2.14,
  "source_text": "...",
  "translation": "..."
}
```

Timeline alignment should always be bound to `id`, not recalculated from translated text.

---

## 13) Control Readable Subtitle Length

After translation, especially into Chinese, subtitles often become:

- too long for one line
- too long for two lines
- too dense to read in time

So before exporting, apply **subtitle fitting**.

### Common rules

- no more than 16 to 22 Chinese characters per line
- maximum two lines
- keep characters per second within a readable range

You can estimate CPS like this:

```python
def cps(text: str, start: float, end: float) -> float:
    duration = max(end - start, 0.1)
    return len(text) / duration
```

### Practical target

- Chinese subtitles should ideally stay below **12 to 15 CPS**
- If too high:
  1. lightly compress wording
  2. then split into two lines
  3. avoid changing timing unless subtitle retiming is allowed

---

## 14) Break Lines, Not Time Segments

Example original timing:

```text
00:00:01,000 --> 00:00:03,500
This is a record loss for our military forces.
```

Translated output:

```text
This is a record loss
for our military forces.
```

Important:

- **only insert line breaks**
- **do not split into new timed subtitle items**

This is the most stable approach.

---

# 7. Automatic Retry Strategy

## 15) Do Not Retry With the Exact Same Prompt Every Time

Retries should be layered.

### First attempt

Normal translation prompt

### Second attempt

Stricter prompt

```text
Translate every item completely.
Do not omit any content.
Keep ids unchanged.
Return JSON only.
```

### Third attempt

Reduce batch size

For example, drop from 5 items to 1

This is especially effective against omissions.

---

## 16) Classify Error Types

Recommended error categories:

- `parse_error`
- `missing_id`
- `extra_output`
- `empty_translation`
- `suspicious_length`
- `entity_loss`
- `untranslated_output`

This makes logs much clearer and helps future prompt tuning.

---

# 8. Recommended Engineering Structure

## 17) Suggested Data Structure

```python
from dataclasses import dataclass

@dataclass
class SubtitleItem:
    id: int
    start: float
    end: float
    source_text_raw: str
    source_text_norm: str
    translation: str = ""
    status: str = "pending"
    error: str = ""
```

---

## 18) Suggested Module Split

```text
asr.py
segment_normalizer.py
translator.py
validator.py
subtitle_formatter.py
exporter.py
pipeline.py
```

### Responsibilities

- `asr.py`: `faster-whisper` transcription
- `segment_normalizer.py`: text cleanup
- `translator.py`: LLM translation calls
- `validator.py`: structural and semantic validation
- `subtitle_formatter.py`: line breaking and length control
- `exporter.py`: export SRT / VTT / JSON
- `pipeline.py`: connect the full workflow

---

# 9. Recommended Execution Strategy

## 19) Best-Practice Configuration

### Stability-first setup

- ASR: `faster-whisper-large-v3`
- Translation: local LLM
- batch size: 3
- temperature: 0
- output format: JSON
- automatically retry failed batches as single items

### Why this works

- large batches are more likely to omit content
- single-item translation is slower
- around 3 items is usually a good balance between stability and speed

---

# 10. A Complete Practical Workflow

```text
1. Use faster-whisper to extract timestamped subtitle segments
2. Clean each segment while preserving both raw and normalized text
3. Group every 3 items into one batch
4. Translate using a strict JSON prompt
5. Validate:
   - valid JSON
   - complete IDs
   - no empty translations
   - no likely untranslated output
   - no abnormal length ratio
   - no key entity loss
6. Retry if validation fails
7. If repeated failures occur, fall back to single-item translation
8. After successful translation, apply line breaking and CPS checks
9. Export srt / vtt
10. Record logs for future prompt tuning
```

---

# 11. Simplified Pseudocode

```python
items = asr_transcribe(audio_path)

items = [normalize_item(x) for x in items]

batches = make_batches(items, batch_size=3)

results = []

for batch in batches:
    translated = translate_batch(batch)

    if not valid_structure(batch, translated):
        translated = retry_translate(batch, mode="strict")

    if not valid_content(batch, translated):
        translated = retry_translate(batch, mode="strict")

    if not valid_content(batch, translated):
        translated = []
        for item in batch:
            one = translate_batch([item], strict=True)
            translated.extend(one)

    results.extend(merge_result(batch, translated))

results = [fit_subtitle_lines(x) for x in results]

export_srt(results, "output.srt")
```

---

# 12. Core Recommendations

If you already have:

- `faster-whisper`
- a local LLM for translation
- subtitle segmentation logic

then the most valuable additions are not “a stronger model,” but these four mechanisms:

### Highest priority

1. **JSON output with IDs**
2. **ID completeness validation**
3. **key entity loss detection**
4. **single-item retry after failure**

These four alone should noticeably improve quality.

### Second priority

5. **length / CPS checks**
6. **line-breaking optimization**
7. **back-translation validation**

---

If you want, I can also turn this into:
1. polished Markdown for [road_map/IMPROVE.md](/home/wilson/work_space/video-caption-assistant/road_map/IMPROVE.md), or
2. a shorter “project-note” version for your repo docs.