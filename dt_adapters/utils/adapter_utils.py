import os
import yaml
from pprint import pprint
from omegaconf import OmegaConf
from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac
from transformers.adapters.configuration import DynamicAdapterFusionConfig
from transformers.adapters.layer import AdapterLayer


def load_adapter_library(adapter_library_file):
    """
    Adapter library is a yaml file that maps adapter names to checkpoint paths and information about
    pretrained adapters.
    """

    # create file if it doesn't exist
    if not os.path.exists(adapter_library_file):
        f = open(adapter_library_file, "w")
        f.close()

    with open(adapter_library_file, "r") as f:
        try:
            adapter_library = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

        if adapter_library is None:
            adapter_library = {}

        print("-" * 50)
        print(f"{len(adapter_library)} adapters available: ")

        for adapter_name, metadata in adapter_library.items():
            print(f"{adapter_name} | best eval score: {metadata['best_eval_score']}")

        print("-" * 50)

    return adapter_library


def insert_new_adapter(adapter_library, model, adapter_name, adapter_config):
    """
    Inserts a new adapter into the model and freezes all other weights.
    """
    # also add an adapter for the new task
    if adapter_name in adapter_library:
        print(
            f"Trained adapter already exists for: {adapter_name}, will be overwriting."
        )

    print(f"inserting new adapter for: {adapter_name} and freezing other model weights")

    # train a new set of adapter weights
    adapter_config["nonlinearity"] = None
    adapter_config["reduction_factor"] = None
    adapter_config = AdapterConfig.load(**adapter_config)
    model.transformer.add_adapter(adapter_name, config=adapter_config, set_active=True)

    # freeze all model weights except of those of this adapter
    model.transformer.train_adapter([adapter_name])

    # set the adapters to be used in every forward pass
    model.transformer.set_active_adapters(adapter_name)


def unfreeze_new_adapter(layer, adapter_name):
    if type(layer) == AdapterLayer:
        if adapter_name in layer.adapters:
            for param in layer.adapters[adapter_name].parameters():
                param.requires_grad = True


def insert_new_fusion_layer(
    adapter_library, model, new_adapter_name, config, adapters_to_use=[]
):
    """
    Adds an adapter for the new task and a fusion layer that fuses the new adapter with the pretrained adapters.

    New adapter is trainable and pretrained adapters are frozen along with the backbone.
    """

    # initialize fusion configuration
    base_fusion_config = dict(DynamicAdapterFusionConfig())
    base_fusion_config.update(OmegaConf.to_container(config.fusion))

    # load adapters to use
    pretrained_adapters_available = list(adapter_library.keys())
    if len(adapters_to_use) == 0 and len(pretrained_adapters_available) == 0:
        print("No pretrained adapters available, stopping program")
        exit()

    if len(adapters_to_use) == 0:
        print("Did not specify adapters to use, using all available adapters")
        adapters_to_use = pretrained_adapters_available

    print("loading pretrained adapter weights...")

    # check that all adapters exist
    for pretrained_adapter_name in adapters_to_use:
        if not pretrained_adapter_name in adapter_library:
            raise Exception("{pretrained_adapter_name} not a valid adapter")

    # load pretrained adapters
    for pretrained_adapter_name in adapters_to_use:
        adapter_ckpt_path = adapter_library[pretrained_adapter_name]["ckpt_path"]

        if not os.path.exists(adapter_ckpt_path):
            raise Exception(f"{adapter_ckpt_path} does not exist")
            exit()

        print(f"Loading {pretrained_adapter_name} from {adapter_ckpt_path}")
        pretrained_adapter = model.transformer.load_adapter(adapter_ckpt_path)

    # add the new trainable adapter
    all_single_task_adapters = [new_adapter_name] + adapters_to_use
    adapter_config = config.adapter
    adapter_config["nonlinearity"] = None
    adapter_config["reduction_factor"] = None
    adapter_config = AdapterConfig.load(**adapter_config)
    model.transformer.add_adapter(
        new_adapter_name, config=adapter_config, set_active=True
    )

    # set the fusion layer as active
    fusion_layer = ac.Fuse(*all_single_task_adapters)
    model.transformer.add_adapter_fusion(
        fusion_layer, config=base_fusion_config, set_active=True
    )

    # training both a new adapter and the fusion layer
    model.transformer.set_active_adapters([fusion_layer, *all_single_task_adapters])

    # make sure all the other weights are frozen except fusion layer and new adapter
    model.transformer.train_adapter_fusion(fusion_layer)

    # make sure the new adapter weights are trainable
    model.transformer.apply_to_adapter_layers(
        lambda i, layer: unfreeze_new_adapter(layer, new_adapter_name)
    )

    # sanity checks to make sure that previous adapter weights are frozen
    for pretrained_adapter_name in adapters_to_use:
        requires_grad = False
        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapters[pretrained_adapter_name]
            .adapter_up.weight.requires_grad
            == requires_grad
        )

    # check that the new adapter is trainable
    assert (
        model.transformer.transformer.h[0]
        .output_adapters.adapters[new_adapter_name]
        .adapter_up.weight.requires_grad
        == True
    )

    # check the the fusion layer is trainable
    assert (
        model.transformer.transformer.h[0]
        .output_adapters.adapter_fusion_layer[fusion_layer.name]
        .query.weight.requires_grad
        == True
    )


def update_adapter_library(adapter_library_file, adapter_name, ckpt_dir, metadata):
    with open(adapter_library_file, "r") as f:
        try:
            adapter_library = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

        if adapter_library is None:
            adapter_library = {}

        new_adapter_info = {
            "name": adapter_name,
            "ckpt_path": str(ckpt_dir),  # where is this adapter file stored
            **metadata,
        }

        # insert new adapter into library
        adapter_library[adapter_name] = new_adapter_info

    # overwrite the file
    with open(adapter_library_file, "w") as f:
        yaml.safe_dump(adapter_library, f)


def save_adapters(model, ckpt_dir, use_fusion=False, adapters=[], metadata=None):
    if use_fusion:
        # save the fusion layer and each individual adapter
        model.transformer.save_adapter_fusion(ckpt_dir, adapters)
        model.transformer.save_all_adapters(ckpt_dir)
    else:
        # save just the adapter weights
        model.transformer.save_adapter(
            ckpt_dir,
            adapters,
            meta_dict=metadata,
        )
