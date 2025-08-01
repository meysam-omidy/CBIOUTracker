from track import Track
from track_state import STATE_UNCONFIRMED, STATE_NEW, STATE_TRACKING, STATE_LOST, STATE_DELETED
import numpy as np
from utils import match_bboxes, tlbr_to_xywh, tlbr_to_tlwh, select_indices
from pydantic import BaseModel

class CBIOUTrackerConfig(BaseModel):
    max_age : int = 30
    min_box_area : int = 10
    max_aspect_ratio : float = 1.6
    high_score_det_threshold : float = 0.5
    low_score_det_threshold : float = 0.1
    init_track_score_threshold : float = 0.6
    buffer_scale1 : float = 0
    buffer_scale2 : float = 0
    buffer_scale3 : float = 0
    # buffer_scale1 : float = 0.3
    # buffer_scale2 : float = 0.5
    # buffer_scale3 : float = 0.5
    match_high_score_dets_with_confirmed_trks_threshold : float = 0.2
    match_low_score_dets_with_confirmed_trks_threshold : float = 0.5
    match_remained_high_score_dets_with_unconfirmed_trks_threshold : float = 0.3
    min_frames_to_predict : int = 3


class CBIOUTracker:
    def __init__(self, config:dict={}):
        self.config = CBIOUTrackerConfig.model_validate(config)
        Track.init(self.config.max_age, self.config.min_box_area, self.config.max_aspect_ratio, self.config.min_frames_to_predict)

    def update(self, boxes): # tlwh
        Track.predict_all()
        high_confidence_detections = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, :4]
        high_scores = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, 4]
        low_confidence_detections = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, :4]
        low_scores = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, 4]
        confirmed_tracks = Track.get_tracks(included_states=[STATE_NEW, STATE_TRACKING, STATE_LOST])
        confirmed_trks = [t.xywh for t in confirmed_tracks] 
        high_confidence_dets = [tlbr_to_xywh(detection) for detection in high_confidence_detections]
        matches, unmatched_confirmed_track_indices, unmatched_high_confidence_detection_indices = match_bboxes(confirmed_trks, high_confidence_dets, self.config.match_high_score_dets_with_confirmed_trks_threshold, self.config.buffer_scale1)
        for track_index, detection_index in matches:
            confirmed_tracks[track_index].update(tlbr_to_tlwh(high_confidence_detections[detection_index]), score=high_scores[detection_index])
        
        remained_confirmed_tracks = select_indices(confirmed_tracks, unmatched_confirmed_track_indices)
        remained_not_lost_tracks = [t for t in remained_confirmed_tracks if t.state in [STATE_TRACKING, STATE_NEW]]
        remained_not_lost_trks = [t.xywh for t in remained_not_lost_tracks]
        low_confidence_dets = [tlbr_to_xywh(detection) for detection in low_confidence_detections]
        matches, unmatched_remained_not_lost_track_indices, unmatched_low_score_detection_indices = match_bboxes(remained_not_lost_trks, low_confidence_dets, self.config.match_low_score_dets_with_confirmed_trks_threshold, self.config.buffer_scale2)
        for track_index, detection_index in matches:
            remained_not_lost_tracks[track_index].update(tlbr_to_tlwh(low_confidence_detections[detection_index]), score=low_scores[detection_index])
        
        remained_high_confidence_detections = select_indices(high_confidence_detections, unmatched_high_confidence_detection_indices)
        remained_high_scores = select_indices(high_scores, unmatched_high_confidence_detection_indices)
        unconfirmed_tracks = Track.get_tracks(included_states=[STATE_UNCONFIRMED])
        unconfirmed_trks = [t.xywh for t in unconfirmed_tracks]
        remained_high_confidence_dets = [tlbr_to_xywh(detection) for detection in remained_high_confidence_detections]
        matches, unmatched_unconfirmed_track_indices, unmatched_remained_high_score_detection_indices = match_bboxes(unconfirmed_trks, remained_high_confidence_dets, self.config.match_remained_high_score_dets_with_unconfirmed_trks_threshold, self.config.buffer_scale3)
        for track_index, detection_index in matches:
            unconfirmed_tracks[track_index].update(tlbr_to_tlwh(remained_high_confidence_detections[detection_index]), score=remained_high_scores[detection_index])
        
        unmatched_remained_high_score_detections = select_indices(remained_high_confidence_detections, unmatched_remained_high_score_detection_indices)
        unmatched_remained_high_scores = select_indices(remained_high_scores, unmatched_remained_high_score_detection_indices)
        for detection, score in zip(unmatched_remained_high_score_detections, unmatched_remained_high_scores):
            if score < self.config.init_track_score_threshold:
                continue
            if Track.FRAME_NUMBER == 1:
                Track(tlbr_to_tlwh(detection), score=score, state=STATE_NEW)
            else:
                Track(tlbr_to_tlwh(detection), score=score)
