import numpy as np
import textwrap
from track_state import STATE_UNCONFIRMED, STATE_NEW, STATE_TRACKING, STATE_LOST, STATE_DELETED, TrackState

class Track:
    @classmethod
    def init(cls, max_age, min_box_area, max_aspect_ratio, min_frames_to_predict):
        cls.INSTANCES:list['Track'] = []
        cls.ID_COUNTER = 1
        cls.FRAME_NUMBER = 0
        cls.MAX_AGE = max_age
        cls.MIN_BOX_AREA = min_box_area
        cls.MAX_ASPECT_RATIO = max_aspect_ratio
        cls.MIN_FRAMES_TO_PREDICT = min_frames_to_predict

    @classmethod
    def predict_all(cls) -> None:
        cls.FRAME_NUMBER += 1
        for track in cls.INSTANCES:
            if track.state not in [STATE_DELETED]:
                track.predict()

    @classmethod
    def get_tracks(cls, included_states:list[TrackState]) -> list['Track']:
        return [track for track in cls.INSTANCES if track.state in included_states]
    
    @property
    def mot_format(self):
        return f"{int(Track.FRAME_NUMBER)},{int(self.id)},{round(self.bbox[0], 1)},{round(self.bbox[1], 1)},{round(self.bbox[2], 1)},{round(self.bbox[3], 1)},{round(self.score, 2)},-1,-1,-1"

    @property
    def clean_format(self):
        return textwrap.dedent(f"""
            **************************************************************************************************************
            id         -> {self.id}
            state      -> {self.state.name}
            bbox       -> {self.bbox}
            age        -> {self.age}
            score      -> {self.score}
            entered    -> {self.entered_frame}
            {f'exited     -> {self.exited_frame}' if self.state == STATE_DELETED else ''}
            {f'last state -> {self.last_state.name}' if self.last_state else ''}
            """).strip()
    
    @property
    def compressed_format(self):
        return f"{self.state.name}    {self.id}    {self.bbox}    {self.age}    {self.score}    {self.entered_frame}    {self.exited_frame}    {self.last_state.name if self.last_state else ''}"

    @property
    def score(self):
        if len(self.scores) > 0:
            # return float(np.mean(self.scores))
            return self.scores[-1]
        else:
            return 0

    @property
    def tlbr(self):
        bbox = self.bbox
        return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    @property
    def xywh(self):
        bbox = self.bbox
        return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]])
    
    @property
    def valid(self):
        invalid_conditions = [
            self.age > Track.MAX_AGE,
            self.state == STATE_UNCONFIRMED and self.age >= 2,
            (self.bbox[2] * self.bbox[3]) < Track.MIN_BOX_AREA,
            (self.bbox[2] / self.bbox[3]) > Track.MAX_ASPECT_RATIO,
            np.any(np.isnan(self.bbox)) or np.any(self.bbox[2:] <= 0)
        ]   
        if any(invalid_conditions):
            return False
        else:
            return True

    def __init__(self, bbox, score, state=None):
        if state == None:
            self.state = STATE_UNCONFIRMED
        else:
            self.state = state
        self.last_state = None
        self.bbox = np.array(bbox).copy()
        self.predict_history = []
        self.update_history = [np.array(bbox).copy()]
        self.scores = [float(score)]
        self.age = 0
        self.entered_frame = Track.FRAME_NUMBER
        self.exited_frame = -1
        self.id = Track.ID_COUNTER
        Track.ID_COUNTER += 1
        Track.INSTANCES.append(self)

    def __str__(self):
        return self.clean_format

    def predict(self):
        self.age += 1
        if not self.valid:
            self.last_state = self.state
            self.state = STATE_DELETED
            self.exited_frame = Track.FRAME_NUMBER
            return
        if self.state in [STATE_TRACKING, STATE_NEW] and self.age >= 2:
            self.last_state = self.state
            self.state = STATE_LOST
        if STATE_TRACKING in [self.state, self.last_state]:
            delta = (self.update_history[-1] - self.update_history[-Track.MIN_FRAMES_TO_PREDICT]) / (Track.MIN_FRAMES_TO_PREDICT - 1)
            new_bbox = self.predict_history[-1] + delta
            self.predict_history.append(new_bbox)
            self.bbox = new_bbox.copy()
            
    def update(self, bbox, score):
        self.update_history.append(np.array(bbox).copy())
        self.scores.append(float(score))
        self.bbox = np.array(bbox).copy()
        self.age = 0
        if self.state == STATE_NEW and len(self.update_history) >= Track.MIN_FRAMES_TO_PREDICT:
            self.predict_history = self.update_history.copy()
            self.state = STATE_TRACKING
        if self.state == STATE_UNCONFIRMED:
            self.state = STATE_NEW
        if self.state == STATE_LOST:
            self.state = self.last_state
            self.last_state = None