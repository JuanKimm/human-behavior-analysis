"""Boundary tests for ID/pose alignment, sequence isolation and synchronized input."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtmc_fall.contracts import TrackedPose, PoseSequence, SequenceCollector
from mtmc_fall.fall import OfflineFallAnalyzer, select_camera_results
from mtmc_fall.tracking import load_tracker, PosePredictor, TrackingAdapter
from mtmc_fall.video import PairedVideoReader


def pose(frame=0, cam=1, gid=1, local=1, epoch=0, quality=.9):
    return TrackedPose(frame, frame / 30, cam, gid, local, epoch, 0, (1280, 720),
                       [100, 100, 200, 400], .9, np.full((17, 2), 110), np.full(17, quality))


class ArrayTensor:
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float32)
    def cpu(self): return self
    def numpy(self): return self.a


class Boxes:
    def __init__(self, boxes, scores): self.xyxy, self.conf = ArrayTensor(boxes), ArrayTensor(scores)
    def __len__(self): return len(self.conf.a)


class FakeCV2:
    CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = range(4)
    def __init__(self, fps=(30, 30), counts=(3, 3), actual=(3, 3)):
        self.fps, self.counts, self.actual, self.caps = fps, counts, actual, []
    def VideoCapture(self, path):
        i = int(path) - 1; owner = self
        class Cap:
            index = 0
            closed = False
            def isOpened(self): return True
            def get(self, k): return [owner.fps[i], owner.counts[i], 4, 2][k]
            def read(self):
                n = self.index; self.index += 1
                return (True, np.full((2, 4, 3), n, np.uint8)) if n < owner.actual[i] else (False, None)
            def release(self): self.closed = True
        cap = Cap(); self.caps.append(cap); return cap


class IntegrationContracts(unittest.TestCase):
    def test_pose_is_a_snapshot(self):
        xy = np.zeros((17, 2)); p = replace(pose(), keypoints=xy)
        xy[:] = 9
        self.assertTrue((p.keypoints == 0).all())
        with self.assertRaises(ValueError): p.keypoints[0, 0] = 3
        with self.assertRaises(ValueError): replace(p, confidence=np.full(17, np.nan))

    def test_sequences_split_on_gap_local_epoch_and_camera(self):
        c = SequenceCollector(30)
        c.append_frame(0, [pose(0), pose(0, cam=2)])
        c.append_frame(1, [pose(1, cam=2)])
        c.append_frame(2, [pose(2), pose(2, cam=2)])
        c.append_frame(3, [pose(3, local=2)])
        c.append_frame(4, [pose(4, local=2, epoch=1)])
        seq = c.finish()
        self.assertEqual([len(s.poses) for s in seq], [1, 1, 1, 1, 3])
        self.assertEqual(len(c.finish()), len(seq))

    def test_duplicate_and_out_of_order_frames_fail_without_mutating(self):
        c = SequenceCollector(30)
        with self.assertRaises(ValueError): c.append_frame(0, [pose(), pose()])
        c.append_frame(0, [pose()])
        with self.assertRaises(ValueError): c.append_frame(2, [pose(2)])
        c.append_frame(1, [pose(1)])
        self.assertEqual(len(c.finish()[0].poses), 2)

    def test_sequence_rejects_mixed_id_and_time(self):
        with self.assertRaises(ValueError): PoseSequence((pose(), pose(1, gid=2)), 30)
        with self.assertRaises(ValueError): PoseSequence((replace(pose(), timestamp=.1),), 30)

    def test_reader_preserves_frame_order_and_count(self):
        cv = FakeCV2(); reader = PairedVideoReader(('1', '2'), SimpleNamespace, cv)
        packets = [reader.next() for _ in range(3)]
        self.assertIsNone(reader.next()); reader.close()
        self.assertEqual([p.indices for p in packets], [{1: i, 2: i} for i in range(3)])
        self.assertEqual([p.frames[2][0, 0, 0] for p in packets], [0, 1, 2])
        self.assertTrue(all(c.closed for c in cv.caps))

    def test_reader_rejects_mismatch_and_decode_failure(self):
        for cv in (FakeCV2(fps=(30, 25)), FakeCV2(counts=(3, 4))):
            with self.assertRaises(ValueError): PairedVideoReader(('1', '2'), SimpleNamespace, cv)
            self.assertTrue(all(c.closed for c in cv.caps))
        cv = FakeCV2(actual=(3, 2)); reader = PairedVideoReader(('1', '2'), SimpleNamespace, cv)
        reader.next(); reader.next()
        with self.assertRaises(RuntimeError): reader.next()
        reader.close()

    def test_original_detection_filters_preserve_pose_indices_and_partial(self):
        legacy = load_tracker(ROOT)
        boxes = [[5, 5, 6, 6], [100, 100, 200, 400], [101, 100, 201, 400], [310, 100, 410, 400]]
        xy = np.array([np.full((17, 2), n) for n in (5, 110, 210, 310)], np.float32)
        result = SimpleNamespace(boxes=Boxes(boxes, [.99, .9, .8, .9]),
                                 keypoints=SimpleNamespace(xy=ArrayTensor(xy), conf=ArrayTensor(np.full((4, 17), .9))))
        class Model:
            calls = 0
            def predict(self, **kwargs): self.calls += 1; return [result, result]
        model = Model(); wrapper = PosePredictor(model)
        packet = legacy.Packet(0, 0., {c: np.zeros((720, 1280, 3), np.uint8) for c in (1, 2)},
                               {1: 0, 2: 0}, {1: 0., 2: 0.}, {1: True, 2: True})
        trackers = {c: SimpleNamespace(tracked_stracks=[], lost_stracks=[], duplicate_detections_removed=0) for c in (1, 2)}
        def partial(tracker, rows, cfg):
            for row in rows: row.partial = True
        args = SimpleNamespace(imgsz=640, device='cpu', yolo_precision={'half': False})
        with patch.object(legacy, 'prepare_occlusions', partial):
            rows, _ = legacy.detect_batch(wrapper, SimpleNamespace(encode=lambda crops: []), packet,
                legacy.Calibration(), trackers, {1: 0, 2: 0}, args, legacy.Config(), legacy.Timings())
        self.assertEqual(model.calls, 1)
        self.assertEqual([r.detector_index for r in rows[1]], [1, 3])
        labels = {}
        for cam in (1, 2):
            for i, row in enumerate(rows[cam]): row.local_id = i + 1
            labels[cam] = [(r, i + 1, 'test') for i, r in enumerate(rows[cam])]
        adapter = object.__new__(TrackingAdapter)
        adapter.pending = {0: dict(zip((1, 2), wrapper.last_poses))}
        _, poses, _, _ = adapter._convert((packet, labels, .3, False))
        self.assertEqual([p.keypoints[0, 0] for p in poses], [110, 310, 110, 310])
        self.assertTrue(all(p.partial for p in poses))

    def test_lifter_coordinates_confidence_and_absolute_window_mapping(self):
        sequence = PoseSequence(tuple(pose(i) for i in range(20, 100)), 30)
        seen = []
        class Lifter:
            def lift(self, xy, width, height):
                seen.append((xy.copy(), width, height)); return np.zeros((len(xy), 17, 3), np.float32)
        class Engine:
            config = {'pose_quality': {'confidence_threshold': .25}}
            def infer_skeleton_sequence(self, skel, **kwargs):
                seen.append(kwargs)
                result = dict(frame_count=len(skel), fps=30, tcn=[], stds=[], dbn=[
                    dict(window_start=0, window_end=63, label='Normal', status='ok', probabilities=[1, 0, 0]),
                    dict(window_start=16, window_end=79, label='Danger', status='ok', probabilities=[0, 0, 1])])
                return SimpleNamespace(to_dict=lambda: result)
        _, _, timeline = OfflineFallAnalyzer(Lifter(), Engine()).analyze(sequence)
        self.assertEqual(seen[0][1:], (1280, 720)); np.testing.assert_array_equal(seen[0][0], np.full((80, 17, 2), 110))
        np.testing.assert_allclose(seen[1]['confidence'], .9)
        self.assertEqual(timeline[62]['status'], 'warmup')
        self.assertEqual(timeline[63]['window_start'], 20)
        self.assertEqual(timeline[63]['decision_frame'], 83)
        self.assertEqual(timeline[-1]['label'], 'Danger')
        self.assertEqual(timeline[-1]['probabilities'], [0, 0, 1])

    def test_shared_models_reset_dbn_between_people_and_cameras(self):
        from runtime.engine import FallRiskInferenceEngine
        engine = FallRiskInferenceEngine(ROOT, device='cpu')
        class Lifter:
            def lift(self, xy, width, height):
                return np.zeros((len(xy), 17, 3), np.float32)
        analyzer = OfflineFallAnalyzer(Lifter(), engine)
        model_ids = (id(engine.tcn), id(engine.stds), id(engine.dbn.model))
        a = PoseSequence(tuple(pose(i, gid=1, cam=1) for i in range(64)), 30)
        b = PoseSequence(tuple(pose(i, gid=2, cam=2) for i in range(64)), 30)
        _, first, _ = analyzer.analyze(a)
        # A deliberately invalid previous posterior must never reach the next person's DBN.
        engine.dbn.prev_log_post = np.full(3, np.nan)
        _, second, _ = analyzer.analyze(b)
        np.testing.assert_array_equal(first['dbn'][0]['probabilities'], second['dbn'][0]['probabilities'])
        self.assertEqual(model_ids, (id(engine.tcn), id(engine.stds), id(engine.dbn.model)))

    def test_camera_selection_uses_pose_quality_not_risk(self):
        def row(cam, label, valid, mean, status='ok'):
            return dict(frame_index=70, timestamp=70/30, global_id=1, camera_id=cam, label=label,
                        status=status, probabilities=[1, 0, 0], decision_frame=63,
                        pose_quality=dict(valid_joint_ratio=valid, mean_joint_confidence=mean))
        a, b = row(1, 'Normal', 1., .8), row(2, 'Danger', .9, .99)
        selected = select_camera_results([b, a])[0]
        self.assertEqual(selected['selected_camera_id'], 1); self.assertEqual(len(selected['cameras']), 2)
        a['status'] = 'insufficient_pose'; self.assertEqual(select_camera_results([a, b])[0]['selected_camera_id'], 2)
        b['status'] = 'warmup'; self.assertIsNone(select_camera_results([a, b])[0]['label'])
        with self.assertRaises(ValueError): select_camera_results([a, a])


if __name__ == '__main__': unittest.main()
