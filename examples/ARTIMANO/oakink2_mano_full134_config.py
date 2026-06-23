from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)


oakink2_mano_full134_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),

    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["hand"],
    ),

    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),

    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(
    oakink2_mano_full134_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
