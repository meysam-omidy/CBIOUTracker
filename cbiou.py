from track import *
from utils import match_bboxes, tlbr_to_xywh, tlbr_to_tlwh
from pydantic import BaseModel

class CBIOUTrackerConfig(BaseModel):
    max_age : int = 30
    min_box_area : int = 10
    max_aspect_ratio : float = 1.6
    detection_threshold : float = 0.6
    buffer_scale1:float = 0.3
    buffer_scale2:float = 0.5
    match_threshold1:float = 0.2
    match_threshold2:float = 0.5
    min_frames_to_predict:int = 3


class CBIOUTracker:
    def __init__(self, config:dict={}):
        self.config = CBIOUTrackerConfig.model_validate(config)
        Track.init(self.config.max_age, self.config.min_box_area, self.config.max_aspect_ratio, self.config.min_frames_to_predict)

    def update(self, boxes): # tlwh
        Track.predict_all()
        detections = boxes[boxes[:, 4] > self.config.detection_threshold][:, :4]
        scores = boxes[boxes[:, 4] > self.config.detection_threshold][:, 4]
        tracks = Track.ALIVE_TRACKS
        dets = [tlbr_to_xywh(detection) for detection in detections]
        trks = [track.xywh for track in tracks]
        matches, unmatched_track_indices, unmatched_detection_indices = match_bboxes(trks, dets, self.config.match_threshold1, self.config.buffer_scale1)
        for track_index, detection_index in matches:
            tracks[track_index].update(tlbr_to_tlwh(detections[detection_index]), score=scores[detection_index])
        remained_detections = [detections[unmatched_detection_index] for unmatched_detection_index in unmatched_detection_indices]
        remained_scores = [scores[unmatched_detection_index] for unmatched_detection_index in unmatched_detection_indices]
        remained_tracks = [tracks[unmatched_track_index] for unmatched_track_index in unmatched_track_indices]
        remained_dets = [tlbr_to_xywh(remained_detection) for remained_detection in remained_detections]
        remained_trks = [remained_track.xywh for remained_track in remained_tracks]
        matches, unmatched_track_indices, unmatched_detection_indices = match_bboxes(remained_trks, remained_dets, self.config.match_threshold2, self.config.buffer_scale2)
        for track_index, detection_index in matches:
            remained_tracks[track_index].update(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index])
        for detection_index in unmatched_detection_indices:
            if Track.FRAME_NUMBER == 1:
                Track(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index], state=STATE_NEW)
            else:
                Track(tlbr_to_tlwh(remained_detections[detection_index]), score=remained_scores[detection_index])
