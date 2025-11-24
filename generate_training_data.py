from openai import OpenAI
import openai 
from pydantic import BaseModel, ValidationError
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv

load_dotenv() 

DATA_OF_CHOICE = 'activities'
raw = pd.read_parquet(f'data/501c3_charity_geocoded_{DATA_OF_CHOICE}_clean.parquet', engine='pyarrow')
print(raw.columns)
missions = raw.CONCATENATED_ACTIVITY.dropna().tolist()
missions_subset = missions[0:3_000]

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

classifier_prompt = '''You are classifying nonprofit mission statements as RELIGIOUS (1) or NON-RELIGIOUS (0).

Use these definitions, which reflect experiences of scholars of economics of religion:

- Label 1 (RELIGIOUS) if:
  - The mission mentions religion, faith, God, Christ, Jesus, Bible, gospel, church, ministry, spiritual worship, or similar concepts; OR
  - The mission emphasizes positive community, compassion, hope, moral uplift, or similar concepts
    in a way typical of faith-based charities. Examples: "positive community", "give hope to the poor", "serve our neighbors",
    "acts of compassion", "uplift our community", etc.

- Label 0 (NON-RELIGIOUS) if:
  - The mission clearly focuses on secular activities (healthcare, sports, arts, environment, economic development,
    education, fairs, libraries, recreation, etc.) and does not show an obviously religious or faith-based motivation.

Examples (mission → label):

- "positive community" → 1
- "Christian worship to create positive community impact" → 1
- "religious services" → 1
- "under our faith in god" → 1
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

output_path = f"data/classified_{DATA_OF_CHOICE}_gpt4omini_PROMPT2.csv"
checkpoint_every = 100  # save every 100 items
max_retries = 5 # try each organization x5, if not, continue

# ----- Resume from existing file if available -----
results = []
start_index = 0

# TODO: Review logic of this code. It helps with resuming the annotation from the last available example
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    results = existing_df.to_dict(orient="records")
    start_index = len(existing_df)
    print(f"Resuming from index {start_index}, already have {start_index} rows.")

# take missions_subset
total = len(missions_subset)

for i in range(start_index, total):
    mission = missions_subset[i] # take a specific mission
    prompt = classifier_prompt.format(mission_text=mission) # format the prompt

    # retry loop
    for attempt in range(max_retries): # keep in mind max retries (5)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # model of choice
                messages=[
                    {"role": "system", "content": "You are a careful JSON-only classifier."}, # give the model a role
                    {"role": "user", "content": prompt}, # the user enters the prompt
                ],
                response_format={"type": "json_object"}, # structure the output as a json object
                temperature=0, # no creativity
            )

            raw_output = response.choices[0].message.content.strip() # get the output

            # Parse JSON + validate with Pydantic
            try:
                data = json.loads(raw_output) # parse the data as a json format
                structured = MissionLabel(**data)
            except (json.JSONDecodeError, ValidationError): # account for errors potentially
                print(f"[Warning] Malformed output on item {i}, keeping raw text.")
                structured = MissionLabel(label=None, reason=raw_output)

            results.append({ # append results to list
                "mission": mission, 
                "label": structured.label, # 0/1 label
                "reason": structured.reason,
            })

            print(f"[{i+1}/{total}] {mission[:50]}... → {structured.label}") # progress output

            # NOTE: small sleep to be nice to the API and avoid errors
            time.sleep(0.2)
            break  # success, exit retry loop

        except openai.RateLimitError as e: # if rate limit API error, then wait
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
    # i starts at 0 for the first item and finishes when reaching the final one - saves items every 100
    if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False) # save data
        print(f"Checkpoint saved at {i+1} items → {output_path}")

# final save after loop
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print("Finished. Final results saved.")
