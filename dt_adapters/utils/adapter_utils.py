import os
import yaml
from pprint import pprint
from omegaconf import OmegaConf
from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac
from transformers.adapters.configuration import DynamicAdapterFusionConfig
from transformers.adapters.layer import AdapterLayer
from transformers.adapters.modeling import NICECouplingBlock
from typing import Iterable, List, Optional, Tuple, Union


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


def unfreeze_new_adapter(layer, adapter_name=None):
    if type(layer) == AdapterLayer:
        if adapter_name in layer.adapters:
            for param in layer.adapters[adapter_name].parameters():
                param.requires_grad = True
    if type(layer) == NICECouplingBlock:
        for param in layer.parameters():
            param.requires_grad = True


def load_adapter(
    model,
    adapter_library=[],
    adapter_keys_to_use=[],
    adapter_key="",
    adapter_ckpt_path=None,
):
    if adapter_key:
        if adapter_key not in adapter_library:
            raise Exception(f"{adapter_key} not in adapter library")
            exit()

        # make sure we are using the model with best checkpoint
        adapter_ckpt_path = adapter_library[adapter_key]["ckpt_path"]
        base_path = os.path.dirname(adapter_ckpt_path)
        best_epoch = adapter_library[adapter_key]["best_eval_epoch"]
        adapter_name = adapter_library[adapter_key]["name"]
        adapter_ckpt_path = os.path.join(base_path, f"epoch_{best_epoch:04d}")

        print(f"Loading {adapter_name} from {adapter_ckpt_path}")
        pretrained_adapter = model.transformer.load_adapter(adapter_ckpt_path)

        # set active so the adapters are used during the forward pass
        model.transformer.set_active_adapters(adapter_name)
        return adapter_name, adapter_ckpt_path
    else:
        # load pretrained adapters from library
        adapters_names, adapter_ckpt_paths = [], []
        for pretrained_adapter_key in adapter_keys_to_use:
            adapter_name, adapter_ckpt_path = load_adapter(
                model, adapter_library, adapter_key=pretrained_adapter_key
            )
            adapters_names.append(adapter_name)
            adapter_ckpt_paths.append(adapter_ckpt_path)

        model.transformer.set_active_adapters(adapters_names)
        return adapters_names, adapter_ckpt_paths


def load_fusion_layer(model, adapter_library, adapter_keys_to_use=[], task_name=""):
    # load pretrained adapters
    load_adapter(model, adapter_library, adapter_keys_to_use)
    fusion_layer_key = (
        f"{task_name}" + "," + ",".join(adapter_keys_to_use) + "_{exp_name}"
    )
    print("Loading fusion layer: ", fusion_layer_key)

    if fusion_layer_key not in adapter_library:
        raise Exception(f"{fusion_layer_key} not in adapter library")
        exit()

    fusion_layer_path = adapter_library[fusion_layer_key]["ckpt_path"]
    model.transformer.load_adapter_fusion(fusion_layer_path, set_active=True)


def insert_new_fusion_layer(
    adapter_library, model, new_adapter_name, config, adapter_keys_to_use=[]
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
    if len(adapter_keys_to_use) == 0 and len(pretrained_adapters_available) == 0:
        print("No pretrained adapters available, stopping program")
        exit()

    if len(adapter_keys_to_use) == 0:
        print("Did not specify adapters to use, using all available adapters")
        adapter_keys_to_use = pretrained_adapters_available

    print("loading pretrained adapter weights...")

    # check that all adapters exist
    for pretrained_adapter_name in adapter_keys_to_use:
        if not pretrained_adapter_name in adapter_library:
            raise Exception(f"{pretrained_adapter_name} not a valid adapter")

    # load pretrained adapters
    adapters_names, adapter_ckpt_paths = load_adapter(
        model,
        adapter_library=adapter_library,
        adapter_keys_to_use=adapter_keys_to_use,
    )

    # add the new trainable adapter
    if base_fusion_config["add_new_unfrozen_adapter"] is True:
        print(f'Adding new Adapter {new_adapter_name}')
        all_single_task_adapters = [new_adapter_name] + adapters_names
        adapter_config = config.adapter
        adapter_config["nonlinearity"] = None
        adapter_config["reduction_factor"] = None
        adapter_config = AdapterConfig.load(**adapter_config)
        model.transformer.add_adapter(
            new_adapter_name, config=adapter_config, set_active=True
        )
        # initialize new adapter from a pretrained adapter
        # randomly choose an adapter to use
        if config.adapter_init_strategy != "none":
            if config.adapter_init_strategy == "random":
                indx = np.random.randint(len(adapters_names))
            elif config.adapter_init_strategy == "language":
                raise NotImplementedError
            else:
                indx = adapters_names.index(config.adapter_init_strategy)
            
            adapter_to_initialize_from = adapters_names[indx]
            adapter_to_initialize_from_ckpt_path = adapter_ckpt_paths[indx]
            
            print(f'initializing new adapter {new_adapter_name} from pretrained: {adapter_to_initialize_from} ...')
            initialized_new_adapter = model.transformer.load_adapter(
                adapter_to_initialize_from_ckpt_path, 
                load_as=new_adapter_name, 
                set_active=True
            )
    else:
        print(f"No new Adapters added, only existing Adapters {','.join(adapters_names)} used")
        all_single_task_adapters = adapters_names

    # set the fusion layer as active
    fusion_layer = ac.Fuse(*all_single_task_adapters)
    model.transformer.add_adapter_fusion(
        fusion_layer, config=base_fusion_config, set_active=True
    )

    # If fusion type is taco-fusion, add TacoFusion module within main model!
    if base_fusion_config["fusion_method"] == "taco-fusion":
        model.add_taco_fusion(model_dim=model.transformer.config.n_embd, adapter_names=all_single_task_adapters, fusion_config=base_fusion_config)

    # activate the adapters
    model.transformer.set_active_adapters([fusion_layer, *all_single_task_adapters])

    # make sure all the other weights are frozen except fusion layer and new adapter
    model.transformer.train_adapter_fusion(fusion_layer)

    if base_fusion_config["add_new_unfrozen_adapter"] is True or base_fusion_config["unfreeze_task_adapter"] is True:
        # make sure the new adapter weights are trainable
        print(f"Making sure new adapter {new_adapter_name} is unfrozen")
        model.transformer.apply_to_adapter_layers(
            lambda i, layer: unfreeze_new_adapter(layer, new_adapter_name)
        )

        # also unfreeze the invertible adapters?
        invertible_layer = model.transformer.transformer.invertible_adapters[
            new_adapter_name
        ]
        unfreeze_new_adapter(invertible_layer)

    # sanity checks to make sure that previous adapter weights are frozen
    for pretrained_adapter_name in adapters_names:
        requires_grad = False
        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapters[pretrained_adapter_name]
            .adapter_up.weight.requires_grad
            == requires_grad
        )

    if base_fusion_config["add_new_unfrozen_adapter"] is True or base_fusion_config["unfreeze_task_adapter"] is True:
        # check that the new adapter is trainable
        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapters[new_adapter_name]
            .adapter_up.weight.requires_grad
            == True
        )

    # check the the fusion layers/weights are trainable
    if base_fusion_config["fusion_method"] == "weighted-composition":
        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapter_fusion_layer[fusion_layer.name]
            .unscaled_weights.requires_grad
            == True
        )
    elif base_fusion_config["fusion_method"] == "bert-fusion":
        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapter_fusion_layer[fusion_layer.name]
            .query.weight.requires_grad
            == True
        )
    elif base_fusion_config["fusion_method"] == "taco-fusion":
        assert (
            model.taco_fusion[fusion_layer.name].W_q.weight.requires_grad
            == True
        )

    return fusion_layer, all_single_task_adapters


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
        adapter_library[
            f"{adapter_name}_{metadata['exp_name']}_{metadata['seed']}"
        ] = new_adapter_info

    # overwrite the file
    with open(adapter_library_file, "w") as f:
        yaml.safe_dump(adapter_library, f)


def save_adapters(
    model, ckpt_dir, use_fusion, adapters: Union[list, str], metadata=None
):
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
