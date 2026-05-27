import unittest

from transformers import AutoConfig

from optimum.exporters.openvino.model_configs import NemotronLabsDiffusionVLMOpenVINOConfig, VLMConfigBehavior
from optimum.exporters.tasks import TasksManager


class NemotronLabsDiffusionVLMExportConfigTest(unittest.TestCase):
    def test_export_config_registration(self):
        model_id = "nvidia/Nemotron-Labs-Diffusion-VLM-8B"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        self.assertEqual(config.model_type, "nemotron_labs_diffusion_vlm")
        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
            "nemotron_labs_diffusion_vlm", exporter="openvino", library_name="transformers"
        )
        self.assertIn("image-text-to-text", supported_tasks)

        export_config = NemotronLabsDiffusionVLMOpenVINOConfig(config=config, task="image-text-to-text")
        self.assertIn("pixel_values", export_config.inputs)
        self.assertIn("image_sizes", export_config.inputs)

        text_embeddings_config = export_config.with_behavior(VLMConfigBehavior.TEXT_EMBEDDINGS)
        language_config = export_config.with_behavior(VLMConfigBehavior.LANGUAGE)

        self.assertIn("input_ids", text_embeddings_config.inputs)
        self.assertIn("inputs_embeds", language_config.inputs)


if __name__ == "__main__":
    unittest.main()
