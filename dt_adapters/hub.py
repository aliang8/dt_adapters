import os
import yaml
import dt_adapters.constants as constants
from pprint import pprint
from omegaconf import OmegaConf
from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac
from transformers.adapters.configuration import DynamicAdapterFusionConfig


class TaskAdapterHub(object):
    def __init__(self, config):
        self.config = config

        # Look at what trained adapters are already available
        with open(constants.HUB_FILE, "r") as f:
            try:
                adapter_info = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

            print("-" * 50)
            self.adapter_library = {a["name"]: a["ckpt_path"] for a in adapter_info}
            print(f"{len(self.adapter_library)} adapters available: ")
            pprint(self.adapter_library)
            print("-" * 50)

    def insert_new_adapter(self, adapter_name, model):
        # also add an adapter for the new task
        if adapter_name in self.adapter_library:
            print(
                f"Trained adapter already exists for: {adapter_name}, will be overwriting."
            )

        print(f"Insert new adapter for: {adapter_name}")

        # train a new set of adapter weights
        adapter_config = self.config.adapter
        adapter_config["nonlinearity"] = None
        adapter_config["reduction_factor"] = None
        adapter_config = AdapterConfig.load(**self.config.adapter)
        model.transformer.add_adapter(
            adapter_name, config=adapter_config, set_active=True
        )

        # freeze all model weights except of those of this adapter
        model.transformer.train_adapter([adapter_name])

        # set the adapters to be used in every forward pass
        model.transformer.set_active_adapters(adapter_name)

    def insert_new_fusion_layer(self, new_adapter_name, model):
        fusion_config = dict(DynamicAdapterFusionConfig())
        fusion_config.update(OmegaConf.to_container(self.config.fusion))

        # For now we only train a new fusion layer
        # maybe consider training a new adapter in addition to fusion
        adapters_to_use = OmegaConf.to_container(self.config.adapters_to_use)

        # load adapters to use
        print("Loading adapter weights...")
        print(f"Adapters to use: {adapters_to_use}")

        # check that all adapters exist
        for adapter_name in adapters_to_use:
            if not adapter_name in self.adapter_library:
                raise Exception("{adapter_name} not a valid adapter")

        for adapter_name in adapters_to_use:
            adapter_ckpt_path = self.adapter_library[adapter_name]
            print(f"Loading {adapter_name} from {adapter_ckpt_path}")
            adapter_name = model.transformer.load_adapter(adapter_ckpt_path)

        # add the new adapter
        adapters_to_use.append(new_adapter_name)

        # set the fusion layer as active
        fusion_layer = ac.Fuse(*adapters_to_use)
        model.transformer.add_adapter_fusion(
            fusion_layer, config=fusion_config, set_active=True
        )
        model.transformer.set_active_adapters([fusion_layer, *adapters_to_use])

        # make sure all the other weights are frozen except fusion layer and new adapter
        model.transformer.train_adapter([new_adapter_name])
        model.transformer.train_adapter_fusion(fusion_layer)

        # check the requires grad
        for adapter_name in adapters_to_use:
            if adapter_name == new_adapter_name:
                requires_grad = True
            else:
                requires_grad = False

            assert (
                model.transformer.transformer.h[0]
                .output_adapters.adapters[adapter_name]
                .adapter_up.weight.requires_grad
                == requires_grad
            )

        assert (
            model.transformer.transformer.h[0]
            .output_adapters.adapter_fusion_layer[fusion_layer.name]
            .unscaled_weights.requires_grad
            == True
        )

    def update_hub(self, adapter_name, ckpt_dir, epoch, best_perf):
        # open the hub file
        with open(constants.HUB_FILE, "r") as f:
            try:
                adapter_info = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

            new_adapter = {
                "name": adapter_name,
                "ckpt_path": str(ckpt_dir),  # where is this adpater file stored
                "epoch": epoch,
                "best_success_rate": best_perf,
            }

            names = [a["name"] for a in adapter_info]

            # insert new adapter into library
            if adapter_name not in names:
                adapter_info.append(new_adapter)
            else:
                index = names.index(adapter_name)
                adapter_info[index] = new_adapter

        # overwrite the file
        with open(constants.HUB_FILE, "w") as f:
            yaml.safe_dump(adapter_info, f)
