from openai import OpenAI
import openai  # for specific exception types
from pydantic import BaseModel, ValidationError
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv

load_dotenv() 

raw = pd.read_parquet('data/501c3_charity_geocoded_missions_clean.parquet', engine='pyarrow')
missions = raw.CANONICAL_MISSION.dropna().tolist()
missions_subset = missions[0:3_000]

client = OpenAI()

classifier_prompt = '''You are classifying nonprofit mission statements as RELIGIOUS (1) or NON-RELIGIOUS (0).

Use these definitions, which reflect the preferences of my organization:

- Label 1 (RELIGIOUS) if:
  - The mission mentions religion, faith, God, Christ, Jesus, Bible, gospel, church, ministry, spiritual growth, or similar concepts; OR
  - The mission is very general but strongly emphasizes community betterment, compassion, serving the poor, or moral uplift,
    in a way typical of faith-based charities. Examples: "community betterment", "help the poor", "serve our neighbors",
    "acts of compassion", "uplift our community", etc.

- Label 0 (NON-RELIGIOUS) if:
  - The mission clearly focuses on secular activities (healthcare, sports, arts, environment, economic development,
    education, fairs, libraries, recreation, etc.) and does not show an obviously religious or faith-based motivation.

Examples (mission → label):

- "community betterment" → 1
- "to provide weekend food for children in need" → 1
- "religious services" → 1
- "to provide soccer instruction to hanover township youth" → 0
- "operate rural public library" → 0
- "the bangor symphony orchestra provides powerful enriching and diverse musical experiences" → 0

Now classify the following mission:

MISSION: "{mission_text}"

Respond ONLY in this exact JSON format:
{{"label": 0 or 1, "reason": "short explanation"}}
'''

class MissionLabel(BaseModel):
    label: int | None
    reason: str

output_path = "classified_missions_gpt4omini.csv"
checkpoint_every = 100  # save every 100 items
max_retries = 5

# ----- Optional: resume from existing file -----
results = []
start_index = 0

if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    results = existing_df.to_dict(orient="records")
    start_index = len(existing_df)
    print(f"Resuming from index {start_index}, already have {start_index} rows.")

# assume missions_subset is your full list of mission strings
total = len(missions_subset)

for i in range(start_index, total):
    mission = missions_subset[i]
    prompt = classifier_prompt.format(mission_text=mission)

    # retry loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a careful JSON-only classifier."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            raw_output = response.choices[0].message.content.strip()

            # Parse JSON + validate with Pydantic
            try:
                data = json.loads(raw_output)
                structured = MissionLabel(**data)
            except (json.JSONDecodeError, ValidationError):
                print(f"[Warning] Malformed output on item {i}, keeping raw text.")
                structured = MissionLabel(label=None, reason=raw_output)

            results.append({
                "mission": mission,
                "label": structured.label,
                "reason": structured.reason,
            })

            print(f"[{i+1}/{total}] {mission[:50]}... → {structured.label}")

            # small sleep to be nice to the API
            time.sleep(0.2)
            break  # success, exit retry loop

        except openai.RateLimitError as e:
            # backoff and retry
            wait = 2 ** attempt  # 1,2,4,8,16...
            print(f"[Rate limit] item {i+1}, attempt {attempt+1}/{max_retries}, waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            # other errors: log, store, and move on
            print(f"[Error on item {i+1}] {e}")
            results.append({"mission": mission, "label": None, "reason": str(e)})
            break  # don't retry non-rate-limit errors

    # checkpointing: save every N items
    if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Checkpoint saved at {i+1} items → {output_path}")

# final save (in case last chunk < checkpoint_every)
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print("✅ Finished. Final results saved.")
