"""Behavioral regression tests; all accounts, uploads and records are disposable."""
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch

from image_io import decode_image
from segmentation.dataset import PairedFolderDataset, get_dataloaders, _random_brightness, preprocess_mask
from segmentation.graph_cut import GraphCutSegmenter, _generate_visualizations
from segmentation.metrics import SegmentationMetrics, dice_score, hausdorff_distance
from data_splits import classification_split_indices, sample_hash


def png_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class ImageAndSegmentationTests(unittest.TestCase):
    def test_classification_splits_remove_holdout_and_duplicate_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = [], []
            for label in range(4):
                for i in range(12):
                    path = root/f"class{label}_{i}.png"
                    Image.new("L", (8, 8), label*40+i).save(path)
                    train.append((str(path), label))
                test.append(train[-1])
            train.append(train[0])
            test.append(test[0])
            ti, vi, ei = classification_split_indices(train, test, .25)
            train_hash = {sample_hash(train[i][0]) for i in ti}
            val_hash = {sample_hash(train[i][0]) for i in vi}
            test_hash = {sample_hash(test[i][0]) for i in ei}
            self.assertFalse(train_hash & val_hash or train_hash & test_hash or val_hash & test_hash)
            self.assertEqual(len(ei), 4)
            self.assertEqual(len(ti)+len(vi), 44)

    def test_corrupt_upload_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            decode_image(b"invalid png bytes")

    def test_bytes_decode_rgb_and_mask(self):
        image = Image.new("L", (32, 32), 255)
        self.assertEqual(decode_image(png_bytes(image)).mode, "RGB")
        self.assertEqual(decode_image(png_bytes(image), "L").mode, "L")

    def test_mask_export_round_trip(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[15:40, 20:45] = 1
        _, _, exported, _ = _generate_visualizations(Image.new("RGB", (64, 64)), mask)
        restored = np.array(decode_image(png_bytes(exported), "L")) > 127
        np.testing.assert_array_equal(restored, mask.astype(bool))
        self.assertEqual(dice_score(restored, mask), 1.0)

    def test_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            SegmentationMetrics(np.ones((4, 4)), np.ones((2, 8)))

    def test_zero_one_ground_truth_masks_preserved(self):
        mask = np.zeros((32, 32), np.uint8)
        mask[5:15, 8:18] = 1
        np.testing.assert_array_equal(preprocess_mask(Image.fromarray(mask), 32), mask)

    def test_boundary_hausdorff_known_translation(self):
        pred = np.zeros((20, 20), np.uint8)
        true = np.zeros_like(pred)
        pred[3, 3] = 1
        true[6, 7] = 1
        self.assertEqual(hausdorff_distance(pred, true), 5.0)

    def test_empty_and_nonfinite_masks_rejected(self):
        for mask in (np.zeros((0, 0)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                SegmentationMetrics(mask)

    def test_missing_masks_not_silently_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, masks = Path(tmp)/"images", Path(tmp)/"masks"
            images.mkdir(); masks.mkdir()
            Image.new("L", (32, 32)).save(images/"a.png")
            with self.assertRaisesRegex(FileNotFoundError, "Missing ground-truth"):
                PairedFolderDataset(images, masks)
            self.assertEqual(PairedFolderDataset(images, masks, allow_missing_masks=True)[0][1].sum().item(), 0)

    def test_validation_does_not_share_training_augmentations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"images").mkdir(); (root/"masks").mkdir()
            gradient = np.tile(np.arange(32, dtype=np.uint8)*8, (32, 1))
            for i in range(20):
                Image.fromarray(gradient).save(root/"images"/f"{i}.png")
                Image.fromarray((gradient > 127).astype(np.uint8)*255).save(root/"masks"/f"{i}.png")
            train, val, test = get_dataloaders(root, fmt="paired", img_size=32, num_workers=0, val_split=.2, test_split=.2)
            self.assertTrue(train.dataset.dataset.augment)
            self.assertFalse(val.dataset.dataset.augment)
            self.assertFalse(test.dataset.dataset.augment)
            torch.testing.assert_close(val.dataset[0][0], val.dataset[0][0])
            with self.assertRaises(ValueError):
                get_dataloaders(root, fmt="paired", val_split=.8, test_split=.5)

    def test_brightness_preserves_negative_intensities(self):
        with patch("segmentation.dataset.random.uniform", return_value=0):
            image = np.array([[-1., -.5, 0., .5, 1.]], np.float32)
            np.testing.assert_array_equal(_random_brightness(image), image)

    def test_graph_cut_methods_and_blank_images(self):
        for method in ("grabcut", "normalized_cut", "both"):
            result = GraphCutSegmenter(method=method, img_size=64).segment(Image.new("RGB", (64, 64)))
            self.assertEqual(result["mask"].shape, (64, 64))
            self.assertEqual(result["tumor_pixels"], 0)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.old_env = {k: os.environ.get(k) for k in ("DATABASE_URL", "UPLOADS_DIR", "ADMIN_PASSWORD")}
        os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(cls.tmp.name)/"tests.db")
        os.environ["UPLOADS_DIR"] = str(Path(cls.tmp.name)/"uploads")
        os.environ["ADMIN_PASSWORD"] = "test-admin-password"
        import app
        cls.app = app
        app.init_db()
        app.create_user("regression@example.test", "test-password", "Test User", "", None)
        app.create_user("other@example.test", "other-password", "Other User", "", None)
        cls.user = app.get_user_by_email("regression@example.test")
        cls.other = app.get_user_by_email("other@example.test")
        cls.admin = app.get_user_by_email("admin@brain.local")
        manifest = json.loads((ROOT/"webtesting_folder/manifest.json").read_text())
        cls.sample = ROOT/"webtesting_folder"/manifest["images"][0]["relative_path"]
        if not cls.sample.exists():
            cls.sample = sorted((ROOT/"Dataset/Testing/glioma").glob("*.jpg"))[0]

    @classmethod
    def tearDownClass(cls):
        cls.app.engine.dispose()
        cls.app.get_database_engine.clear()
        cls.app.setup_db_once.clear()
        for k, v in cls.old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # AppTest may still hold SQLite handles on Windows; TemporaryDirectory retries cleanup.
        cls.tmp.cleanup()

    def make_app(self, user=None):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT/"app.py"), default_timeout=90).run()
        if user:
            at.session_state.auth_user_id = user.id
            at.run()
        self.assertEqual(list(at.exception), [])
        return at

    def upload(self, data, name="web_mri.png"):
        from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
        from streamlit.proto.Common_pb2 import FileURLs
        return UploadedFile(UploadedFileRec("regression", name, "image/png", data), FileURLs())

    def test_login_and_class_guide_render(self):
        at = self.make_app()
        self.assertTrue(any("Login" == b.label for b in at.button))
        self.assertFalse(any("Model load error" in e.value for e in at.error))

    def test_auth_hash_and_wrong_password(self):
        self.assertTrue(self.app.verify_password("test-password", self.user.password_hash))
        self.assertIsNone(self.app.authenticate_user(self.user.email, "wrong"))
        self.assertEqual(self.app.authenticate_user(self.user.email, "test-password").id, self.user.id)

    def test_web_single_upload_heatmap_and_persistence(self):
        uploaded = self.upload(self.sample.read_bytes(), self.sample.name)
        at = self.make_app(self.user)
        def uploader(label, **kwargs):
            return uploaded if kwargs.get("key") == "user_uploader" else []
        with patch("streamlit.file_uploader", side_effect=uploader):
            next(b for b in at.button if b.label == "Predict").click().run()
        self.assertEqual(list(at.exception), [])
        self.assertFalse(any("failed" in e.value.lower() for e in at.error))
        self.assertTrue(any("Prediction:" in s.value for s in list(at.success)+list(at.warning)+list(at.error)))
        self.assertGreater(len(self.app.get_user_predictions(self.user.id)), 0)
        self.assertFalse(any("Heatmap unavailable" in i.value for i in at.info))

    def test_corrupt_upload_does_not_crash_page(self):
        at = self.make_app(self.user)
        bad = self.upload(b"not an image")
        with patch("streamlit.file_uploader", side_effect=lambda label, **kw: bad if kw.get("key") == "user_uploader" else []):
            next(b for b in at.button if b.label == "Predict").click().run()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(any("Cannot read" in e.value for e in at.error))

    def test_batch_failed_files_reported(self):
        at = self.make_app(self.user)
        bad = self.upload(b"bad file")
        with patch("streamlit.file_uploader", side_effect=lambda label, **kw: [bad] if kw.get("key") == "batch_uploader" else None):
            next(b for b in at.button if b.label == "Run Batch Prediction").click().run()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(any("failed files" in w.value for w in at.warning))
        self.assertFalse(any("high-confidence results" in s.value for s in at.success))

    @unittest.skipUnless((ROOT/"webtesting_folder/images").is_dir(), "Run scripts/download_web_test_images.py for web integration testing")
    def test_all_web_images_through_batch_upload_match_offline_evaluation(self):
        import pandas as pd
        manifest = json.loads((ROOT/"webtesting_folder/manifest.json").read_text())
        uploads = [self.upload((ROOT/"webtesting_folder"/r["relative_path"]).read_bytes(), Path(r["relative_path"]).name)
                   for r in manifest["images"]]
        at = self.make_app(self.user)
        with patch("streamlit.file_uploader", side_effect=lambda label, **kw: uploads if kw.get("key") == "batch_uploader" else None):
            next(b for b in at.button if b.label == "Run Batch Prediction").click().run(timeout=180)
        self.assertEqual(list(at.exception), [])
        rows = at.session_state.batch_results
        self.assertEqual(len(rows), 200)
        self.assertTrue(all(r["status"] == "saved" for r in rows))
        expected = pd.read_csv(ROOT/"webtesting_folder/results/web_predictions.csv")
        self.assertEqual([r["predicted_class"] for r in rows], expected["prediction"].tolist())

    def test_segmentation_upload_with_ground_truth(self):
        at = self.make_app(self.user)
        at.radio[0].set_value("Segmentation").run()
        image = decode_image(self.sample)
        uploaded = self.upload(png_bytes(image))
        gt = self.upload(png_bytes(Image.new("L", image.size, 0)), "ground_truth.png")
        def uploader(label, **kw):
            return gt if kw.get("key") == "seg_gt_uploader" else uploaded
        with patch("streamlit.file_uploader", side_effect=uploader):
            next(b for b in at.button if "Run Graph Cut" in b.label).click().run()
        self.assertEqual(list(at.exception), [])
        self.assertEqual(list(at.error), [])
        self.assertTrue(any("Accuracy Evaluation" in h.value for h in at.subheader))

    def test_admin_literal_search_reset_and_partial_date(self):
        import pandas as pd
        self.app.save_prediction(self.user.id, self.sample.name, "glioma", .8, pd.DataFrame({"Class":["glioma"],"Probability":[.8]}), str(self.sample))
        at = self.make_app(self.admin)
        at.text_input(key="filter_email_search").set_value("[").run()
        self.assertEqual(list(at.exception), [])
        next(b for b in at.button if "Reset Filters" in b.label).click().run()
        self.assertEqual(list(at.exception), [])
        self.assertEqual(at.text_input(key="filter_email_search").value, "")
        from datetime import date
        at.date_input(key="filter_date").set_value((date.today(),)).run()
        self.assertEqual(list(at.exception), [])

    def test_prediction_delete_enforces_ownership(self):
        import pandas as pd
        self.app.save_prediction(self.user.id, "ownership.png", "glioma", .8, pd.DataFrame({"Class":["glioma"],"Probability":[.8]}))
        record = self.app.get_user_predictions(self.user.id)[0]
        self.assertFalse(self.app.delete_prediction_record(record.id, self.other.id, False)[0])

    def test_deleting_record_cannot_delete_dataset_or_external_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp)/"protected.png"
            outside.write_bytes(b"protected test fixture")
            self.app.remove_uploaded_image(str(outside))
            self.assertTrue(outside.exists())

    def test_admin_delete_cleans_up_uploaded_files(self):
        self.app.create_user("disposable@example.test", "test-password", "Disposable", "", None)
        user = self.app.get_user_by_email("disposable@example.test")
        path = self.app.store_uploaded_image(user.id, self.upload(self.sample.read_bytes()))
        self.assertTrue(path.exists())
        self.assertTrue(self.app.delete_user_account_by_admin(user.id, self.admin.id)[0])
        self.assertFalse(path.exists())

    def test_saliency_leaves_model_gradients_unmodified(self):
        model = self.app.load_artifacts()[0]
        self.app.build_saliency_overlay(decode_image(self.sample))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
