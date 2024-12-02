import wandb
import pandas as pandas

wandb.init(project="tiny-blimp", name="updated_table")

csv_file = "./wandb_restructured_long_format.csv"
df = pandas.read_csv(csv_file)

table = wandb.Table(dataframe=df)

wandb.log({"Checkpoint Accuracies": table})

wandb.finish()