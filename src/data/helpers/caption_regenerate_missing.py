import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.data.helpers.caption_generator import CaptionGenerator

captions_csv_path = 'captions.csv'
captions_df = pd.read_csv(captions_csv_path)

caption_generator = CaptionGenerator()

# for each combination of anatomy site and abnormality label, print the the number of captions
for anatomy_site in captions_df['anatomy_site'].unique():
    for label in captions_df['abnormality_label'].unique():
        count = len(captions_df[(captions_df['anatomy_site'] == anatomy_site) & (captions_df['abnormality_label'] == label)])
        print(f"Anatomy site: {anatomy_site}, Abnormality label: {label}, Number of captions: {count}")
        
        # if the count is smaller than 20, generate as many captions as needed to get to 20
        if count < 20:
            number_of_captions_to_generate = 20 - count
            print(f"Generating {number_of_captions_to_generate} new captions for {anatomy_site} with label {label}.")

            new_captions = caption_generator.generate_captions(anatomy_site, label, number_of_captions_to_generate)


            
            # create a new dataframe with the new captions
            new_rows = pd.DataFrame([{
                "anatomy_site": anatomy_site,
                "abnormality_label": label,
                "caption": caption
            } for caption in new_captions])
            
            # append the new rows to the existing dataframe
            captions_df = pd.concat([captions_df, new_rows], ignore_index=True)
            print(f"Generated {number_of_captions_to_generate} new captions for {anatomy_site} with label {label}.")


# save the updated captions dataframe to the csv file
captions_df.to_csv('captions_extended_2.csv', index=False)