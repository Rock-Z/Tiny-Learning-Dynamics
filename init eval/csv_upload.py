import wandb
import pandas as pd

# Log in to W&B
wandb.login()

# Initialize a W&B run
run = wandb.init(project='tiny-blimp', job_type='upload_csv')

# Path to your CSV file
csv_file_path = "./wandb_restructured_long_format.csv"

# Create an artifact to upload the CSV
artifact = wandb.Artifact('restructured-data', type='dataset')

# Add the CSV file to the artifact
artifact.add_file(csv_file_path)

# Log the artifact to W&B
run.log_artifact(artifact)

# Finish the W&B run
run.finish()


