import numpy as np
import lap
import time

def count_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        print(f"{int((end-start) * 1000)} ms    {func.__name__}    {kwargs.get('cost_matrix').shape if np.any(kwargs.get('cost_matrix', None)) else ''}")
        return result
    return wrapper

def batch_biou(bbox1, bbox2, buffer_scale):
    bbox1[:, 2:] *= (1 + buffer_scale)
    bbox2[:, 2:] *= (1 + buffer_scale)
    bbox1 = xywh_to_tlbr(bbox1)
    bbox2 = xywh_to_tlbr(bbox2)
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])                                      
        + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)                                              
    return(o) 

def batch_iou(bb1, bb2):
    bb1 = np.expand_dims(bb1, 1)
    bb2 = np.expand_dims(bb2, 0)
    xx1 = np.maximum(bb1[..., 0], bb2[..., 0])
    yy1 = np.maximum(bb1[..., 1], bb2[..., 1])
    xx2 = np.minimum(bb1[..., 2], bb2[..., 2])
    yy2 = np.minimum(bb1[..., 3], bb2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])                                      
        + (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1]) - wh)                                              
    return(o) 

def assignment(cost_matrix):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches.tolist(), unmatched_a.tolist(), unmatched_b.tolist()

def associate(tracks, detections, iou_threshold, buffer_scale):
    if len(tracks) == 0:
        return [], [], [i for i in range(len(detections))]
    elif len(detections) == 0:
        return [], [i for i in range(len(tracks))], []
    
    track_xywhs = np.array([t.xywh for t in tracks])
    track_tlbrs = np.array([t.tlbr for t in tracks])
    detection_xywhs = np.array([tlbr_to_xywh(d) for d in detections])
    cost = 1 - batch_biou(track_xywhs, detection_xywhs, buffer_scale)
    ious = batch_iou(track_tlbrs, detections)
    matched_tracks, unmatched_tracks, unmatched_detections = assignment(cost)
    matchs_to_remove = []
    for i, j in matched_tracks:
        if ious[i,j] <= iou_threshold:
            matchs_to_remove.append([i,j])
            unmatched_tracks.append(i)
            unmatched_detections.append(j)
    for i,j in matchs_to_remove:
        matched_tracks.remove([i,j])
    return matched_tracks, unmatched_tracks, unmatched_detections
    
def select_indices(arr, indices):
    return [arr[index] for index in indices]
    
def tlbr_to_tlwh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0]
    o[..., 1] = bbox[..., 1]
    o[..., 2] = bbox[..., 2] - bbox[..., 0]
    o[..., 3] = bbox[..., 3] - bbox[..., 1]
    return o

def tlbr_to_xywh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    o[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    o[..., 2] = bbox[..., 2] - bbox[..., 0]
    o[..., 3] = bbox[..., 3] - bbox[..., 1]
    return o

def tlwh_to_tlbr(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0]
    o[..., 1] = bbox[..., 1]
    o[..., 2] = bbox[..., 0] + bbox[..., 2]
    o[..., 3] = bbox[..., 1] + bbox[..., 3]
    return o

def tlwh_to_xywh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] + bbox[..., 3] / 2
    o[..., 2] = bbox[..., 2]
    o[..., 3] = bbox[..., 3]
    return o

def xywh_to_tlwh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    o[..., 2] = bbox[..., 2]
    o[..., 3] = bbox[..., 3]
    return o

def xywh_to_tlbr(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    o[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return o

def tlwh_to_z(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] + bbox[..., 3] / 2
    o[..., 2] = bbox[..., 2] / bbox[..., 3]
    o[..., 3] = bbox[..., 3]
    return o

def tlbr_to_z(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    o[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    o[..., 2] = (bbox[..., 2] - bbox[..., 0]) / (bbox[..., 3] - bbox[..., 1])
    o[..., 3] = bbox[..., 3] - bbox[..., 1]
    return o

def z_to_tlwh(z:np.ndarray) -> np.ndarray:
    o = np.zeros_like(z, dtype=float)
    o[..., 0] = z[..., 0] - z[..., 2] * z[..., 3] / 2
    o[..., 1] = z[..., 1] - z[..., 3] / 2
    o[..., 2] = z[..., 2] * z[..., 3]
    o[..., 3] = z[..., 3]
    return o[:4]

def z_to_tlbr(z:np.ndarray) -> np.ndarray:
    o = np.zeros_like(z, dtype=float)
    o[..., 0] = z[..., 0] - z[..., 2] * z[..., 3] / 2
    o[..., 1] = z[..., 1] - z[..., 3] / 2
    o[..., 2] = z[..., 0] + z[..., 2] * z[..., 3] / 2
    o[..., 3] = z[..., 1] + z[..., 3] / 2
    return o[:4]

def z_to_xywh(z:np.ndarray) -> np.ndarray:
    o = np.zeros_like(z, dtype=float)
    o[..., 0] = z[..., 0]
    o[..., 1] = z[..., 1]
    o[..., 2] = z[..., 2] * z[..., 3]
    o[..., 3] = z[..., 3]
    return o[:4]

