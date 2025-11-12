from string import Template
import pandas as pd
from transformers import pipeline
import json
from tqdm import tqdm

class CaptionGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device=-1):
        self.pipe = pipeline("text-generation", model=model_name, device=device)

        self.prompt = """You are a helpful medical assistant. Given the following anatomy site and abnormality label, generate concise and informative captions for a medical image.

I want you to generate a list of %d captions. Make it very diverse and creative but stick to the anatomy site and abnormality label.
The captions should not be longer than 3 sentences and should be written in a professional tone suitable for medical documentation.

You can play with synonyms like (radiograph, x-ray), (abnormal, not normal), (normal, healthy) etc. to make the captions more diverse.

Abnormality label 0 means there is no abnormality, and label 1 means there is an abnormality.


Some examples:
1. "An x-ray image of an elbow with no abnormality."
2. "A radiograph of a forearm showing signs of an abnormality."

Rules:
1. DO NOT WRITE ABOUT INFORMATION THAT IS NOT AVAILABLE TO YOU. THE ONLY THINGS YOU CAN WRITE ABOUT ARE THE ANATOMY SITE AND ABNORMALITY LABEL.
2. DO NOT ADD ANY INFORMATION ABOUT ANYTHING ELSE.
3. ONLY ANSWER IN THE GIVEN JSON FORMAT.

Return a list of captions in the following JSON format:
```json
{{
  "captions": [
    "Caption 1",
    "Caption 2",
    "Caption 3",
    ...
    "Caption %d"
  ]
}}
```
"""
    def generate_captions(self, anatomy_site, abnormality_label, number_of_captions=20):
      this_prompt = self.prompt % (number_of_captions, number_of_captions)
      messages = [
            {"role": "system", "content": this_prompt},
            {"role": "user", "content": f"Anatomy site: {anatomy_site}, Abnormality label: {abnormality_label}"}
        ]

      response = self.pipe(messages, max_new_tokens=2000, num_return_sequences=1, do_sample=True, temperature=0.7)
      
      answer = response[0]['generated_text'][-1]['content']

      # read answer as json
      answer_json = json.loads(answer)
      captions = answer_json['captions']

      return captions
    

if __name__ == "__main__":
  anatomy_sites = ['ELBOW', 'FOREARM', 'ANKLE', 'HUMERUS', 'WRIST', 'FINGER', 'FOOT', 'SHOULDER', 'HAND', 'HIP', 'KNEE']
  abnormality_labels = [0, 1]

  number_of_captions = 20

  captions_df = pd.DataFrame(columns=['anatomy_site', 'abnormality_label', 'caption'])

  captionGenerator = CaptionGenerator()

  with tqdm(total=len(anatomy_sites) * len(abnormality_labels), desc="Captions") as pbar:
    for anatomy_site in anatomy_sites:
      for label in abnormality_labels:
        captions = captionGenerator.generate_captions(anatomy_site, label, 20)
        for caption in captions:
          new_row = pd.DataFrame([{
              "anatomy_site": anatomy_site,
              "abnormality_label": label,
              "caption": caption
          }])

          captions_df = pd.concat([captions_df, new_row], ignore_index=True)

        # Save the captions to a CSV file
        captions_df.to_csv('captions.csv', index=False)

        pbar.update(1)
    