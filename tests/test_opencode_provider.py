import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = json.loads((ROOT / "deployment/opencode-provider.json").read_text())

    def test_all_recipe_models_share_one_local_endpoint(self):
        recipes = [*ROOT.glob("maya/*/recipe.json"), *ROOT.glob("strata/*/recipe.json")]
        expected = {json.loads(p.read_text())["api_model"] for p in recipes}
        self.assertEqual(set(self.provider["models"]), expected)
        self.assertEqual(self.provider["package"], "@opencode/ai/providers/openai-compatible")
        self.assertEqual(self.provider["settings"]["baseURL"], "http://127.0.0.1:8080/v1")
        for model in self.provider["models"].values():
            self.assertEqual(model["limit"]["context"], 524288)
            self.assertEqual(model["limit"]["output"], 8192)
            self.assertTrue(model["capabilities"]["tools"])

    def test_glm_default_and_variants_keep_thinking_controls(self):
        model = self.provider["models"]["glm-5.3-flash"]
        self.assertEqual(model["body"]["reasoning_effort"], "none")
        variants = {v["id"]: v["body"] for v in model["variants"]}
        self.assertFalse(variants["fast"]["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(variants["thinking"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(variants["thinking"]["reasoning_effort"], "high")
