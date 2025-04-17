# deeplabv3plus
This is my submision for the VJT Assignment where I have generated the training mask from json annotations and trained DeepLabv3Plus from scratch in PyTorch for CityScapes Dataset.

[WandB Project Dashboard Link](https://wandb.ai/shishirroy-indian-institute-of-science/vjt_assignment/workspace?nw=nwusershishirroy) - [WandB Report Link](https://api.wandb.ai/links/shishirroy-indian-institute-of-science/19x778h8)

Along with all the segmentation metrics, (image, predicted_mask, gt_mask) for first mini-batch (12 items) of validation set is also logged into a table named "results_table" in WandB where the captions
are in the form {file_type}_{epoch}_{index}.

The trained model weights can be downloaded from [here](https://api.wandb.ai/files/shishirroy-indian-institute-of-science/vjt_assignment/dj2gtusu/deeplabv3plus_epoch_20.pth).
