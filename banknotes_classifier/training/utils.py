from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

callbacks = [ModelCheckpoint(dirpath="./checkpoints", filename="model.ckpt", monitor="accuracy", mode="max"),
             ModelSummary(),
             ]
