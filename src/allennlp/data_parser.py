# from allennlp.commands.train import train_model_from_file
# from allennlp.data import TokenIndexer
# from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
# from allennlp_models.coref import ConllCorefReader

import shutil
import sys

from allennlp.commands import main
from pathlib import Path
config_file = "./trainer.jsonnet"

# Use overrides to train on CPU.

serialization_dir = "../../models/spanbert-senticoref3"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!

# Assemble the command into sys.argv
if __name__ == '__main__':
    Path(serialization_dir).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(serialization_dir, ignore_errors=True)
    sys.argv = [
        "allennlp",
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "custom_conll_reader",
        # "-r"
    ]
    main()
