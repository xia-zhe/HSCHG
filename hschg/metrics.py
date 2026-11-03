"""Official original-order batch32 metric aggregation."""
import numpy as np

def video_metrics(pred, target, classes):
    from .official_metrics import segment_level, event_level
    eye = np.eye(classes, dtype=np.float32)
    (seg, eve) = ([], [])
    for (p, y) in zip(pred, target):
        seg.append(segment_level(eye[p].T, eye[y].T))
        eve.append(event_level(eye[p].T, eye[y].T))
    return (np.asarray(seg, np.float64), np.asarray(eve, np.float64))

def aggregate(pred, target, seg, eve, mask=None, batch=32, bg=67):
    if mask is not None:
        (pred, target, seg, eve) = (pred[mask], target[mask], seg[mask], eve[mask])
    if len(pred) == 0:
        raise ValueError('Empty evaluation subset')
    a_sum = s_sum = e_sum = 0.0
    batches = 0
    for i in range(0, len(pred), batch):
        (p, y) = (pred[i:i + batch], target[i:i + batch])
        n = y.size
        a_sum += float((p == y).sum()) / n * n
        s_sum += np.mean(seg[i:i + batch])
        e_sum += np.mean(eve[i:i + batch])
        batches += 1
    (a, s, e) = (a_sum / target.size, float(s_sum / batches), float(e_sum / batches))
    (back, fore) = (target == bg, target != bg)

    def rate(num, den):
        return float(num / den) if den else None
    return dict(accuracy=a, segment_f1=s, event_f1=e, avg=(a + s + e) / 3, videos=len(pred), f1_batch_count=batches, background_false_positive=rate(((pred != bg) & back).sum(), back.sum()), foreground_false_negative=rate(((pred == bg) & fore).sum(), fore.sum()), foreground_wrong_class=rate(((pred != bg) & (pred != target) & fore).sum(), fore.sum()))

def all_metrics(pred, target, types, classes=68):
    (seg, eve) = video_metrics(pred, target, classes)
    result = {name: aggregate(pred, target, seg, eve, None if name == 'total' else types == typ, bg=classes - 1) for (name, typ) in [('total', None), ('seen', 'close'), ('unseen', 'open')]}
    return (result, seg, eve)
