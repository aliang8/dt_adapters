import os
import yaml
from pprint import pprint
from omegaconf import OmegaConf
from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac
from transformers.adapters.configuration import DynamicAdapterFusionConfig


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
            adapter_info = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

        if adapter_info is None:
            adapter_info = {}

        print("-" * 50)
        adapter_library = {a["name"]: a["ckpt_path"] for a in adapter_info}
        print(f"{len(adapter_library)} adapters available: ")
        pprint(adapter_library)
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


def insert_new_fusion_layer(adapter_library, model, adapter_name, fusion_config):
    fusion_config = dict(DynamicAdapterFusionConfig())
    fusion_config.update(OmegaConf.to_container(fusion_config))

    # for now we only train a new fusion layer
    # maybe consider training a new adapter in addition to fusion
    adapters_to_use = OmegaConf.to_container(adapters_to_use)

    # load adapters to use
    print("loading adapter weights...")
    print(f"adapters to use: {adapters_to_use}")

    # check that all adapters exist
    for adapter_name in adapters_to_use:
        if not adapter_name in adapter_library:
            raise Exception("{adapter_name} not a valid adapter")

    for adapter_name in adapters_to_use:
        adapter_ckpt_path = adapter_library[adapter_name]
        print(f"Loading {adapter_name} from {adapter_ckpt_path}")
        adapter_name = model.transformer.load_adapter(adapter_ckpt_path)

    # add the new adapter
    adapters_to_use.append(adapter_name)

    # set the fusion layer as active
    fusion_layer = ac.Fuse(*adapters_to_use)
    model.transformer.add_adapter_fusion(
        fusion_layer, config=fusion_config, set_active=True
    )
    model.transformer.set_active_adapters([fusion_layer, *adapters_to_use])

    # make sure all the other weights are frozen except fusion layer and new adapter
    model.transformer.train_adapter([adapter_name])
    model.transformer.train_adapter_fusion(fusion_layer)

    # sanity checks to make sure that previous adapter weights are frozen
    for adapter_name in adapters_to_use:
        if adapter_name == adapter_name:
            requires_grad = True
        else:
            requires_grad = False

        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapters[adapter_name]
            .adapter_up.weight.requires_grad
            == requires_grad
        )

    # check the the fusion layer is trainable
    assert (
        model.transformer.transformer.h[0]
        .output_adapters.adapter_fusion_layer[fusion_layer.name]
        .unscaled_weights.requires_grad
        == True
    )


def update_adapter_library(adapter_library_file, adapter_name, ckpt_dir, metadata):
    with open(adapter_library_file, "r") as f:
        try:
            adapter_info = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

        new_adapter_info = {
            "name": adapter_name,
            "ckpt_path": str(ckpt_dir),  # where is this adapter file stored
            **metadata,
        }

        names = [a["name"] for a in adapter_info]

        # insert new adapter into library
        if adapter_name not in names:
            adapter_info.append(new_adapter_info)
        else:
            index = names.index(adapter_name)
            adapter_info[index] = new_adapter_info

    # overwrite the file
    with open(adapter_library_file, "w") as f:
        yaml.safe_dump(adapter_info, f)
