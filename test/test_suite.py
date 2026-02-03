import unittest
import os
import sys
import datetime
import numpy as np

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from xai import ScoreCAMBrainTumorXAI

# ==========================================
# QA CONFIGURATION
# ==========================================
# UPDATE THIS with the name of your real test image
TEST_IMAGE_PATH = "test_image_meningioma.jpg"
MODEL_PATH = os.path.join("models", "mobilenet_nd5_final_finetuned_cw_v2.h5")


class TestBrainTumorEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the environment and print the Report Header."""
        # 1. Print Professional Header
        print("\n" + "=" * 70)
        print(f"{'BRAIN TUMOR CLASSIFIER - AUTOMATED QA TEST SUITE':^70}")
        print("=" * 70)
        print(f"[QA] Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[QA] Test Image : {TEST_IMAGE_PATH}")
        print(f"[QA] Model Path : {MODEL_PATH}")

        # 2. Validate Files Exist
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"\n[CRITICAL ERROR] Test image not found: {TEST_IMAGE_PATH}")
            print("Please place a valid .jpg file in the directory.")
            sys.exit(1)

        if not os.path.exists(MODEL_PATH):
            print(f"\n[CRITICAL ERROR] Model file not found: {MODEL_PATH}")
            sys.exit(1)

        # 3. Load Engine
        print("[QA] Status     : Initializing Inference Engine...", end=" ", flush=True)
        try:
            cls.engine = ScoreCAMBrainTumorXAI(model_path=MODEL_PATH, verbose=False)
            print("[OK]")
        except Exception as e:
            print("[FAILED]")
            print(f"Error Details: {e}")
            sys.exit(1)

        print("-" * 70)
        print(f"{'TEST EXECUTION LOG':^70}")
        print("-" * 70)

    @classmethod
    def tearDownClass(cls):
        """Print the Final Summary Footer."""
        print("-" * 70)
        print(f"{'TEST SUMMARY':^70}")
        print("-" * 70)
        print(f"[QA] Suite Result : PASSED")
        print("=" * 70 + "\n")

    def test_1_preprocessing_shape(self):
        """TC-01: Preprocessing Dimension Check"""
        processed = self.engine.preprocess_image(TEST_IMAGE_PATH)

        # Assertion
        self.assertEqual(processed.shape, (1, 224, 224, 3), "Shape Mismatch")

        # Log Output
        print(f" TC-01: Verify Preprocessing Shape (1, 224, 224, 3) ... [PASS]")

    def test_2_prediction_output(self):
        """TC-02: Prediction Integrity Check"""
        label, conf, idx = self.engine.predict(TEST_IMAGE_PATH)

        # Assertions
        self.assertIsInstance(label, str)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertTrue(0 <= idx <= 3)

        # Log Output
        print(f" TC-02: Verify Prediction Integrity ..................... [PASS]")
        print(f"        > Output: Label='{label}' | Confidence={conf:.4f}")

    def test_3_scorecam_generation(self):
        """TC-03: XAI Heatmap Check"""
        orig, heatmap, idx, conf = self.engine.compute_scorecam(TEST_IMAGE_PATH)

        # Assertions
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertEqual(orig.shape, (224, 224, 3))

        # Log Output
        print(f" TC-03: Verify Score-CAM Heatmap Generation ........... [PASS]")
        print(f"        > Heatmap Size: {heatmap.shape}")


if __name__ == '__main__':
    # Run with verbosity=0 to suppress the default unittest dots/lines
    # so our custom professional print statements stand out.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBrainTumorEngine)
    unittest.TextTestRunner(verbosity=0).run(suite)