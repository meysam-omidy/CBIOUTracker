import os
import pickle
import numpy as np
from cbiou import CBIOUTracker
from track import Track
from track_state import STATE_UNCONFIRMED, STATE_TRACKING, STATE_LOST, STATE_DELETED, TrackState
from evaluate import evaluate
from utils import count_time

@count_time
def run():
    seqs = ['MOT17-13-FRCNN', ]
    # seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ]


    os.makedirs('outputs/cbiou', exist_ok=True)

    seqmap = open('./trackeval/seqmap/mot17/custom.txt', 'w')
    seqmap.write('name\n')
    for seq in seqs:
        
        seqmap.write(f'{seq}\n')
        file = open(f'outputs/cbiou/{seq}.txt', 'w')
        # detections = detections_file[seq]
        detections = np.loadtxt(f'detections/bytetrack_x_mot17/{seq}.txt', delimiter=',')
        # detections = np.loadtxt('detections/MOT17-04-bytetrack-mot17-x.txt', delimiter=',')
        gt_dets_file = np.loadtxt(f'../../.Datasets/MOT17/train/{seq}/gt/gt.txt', delimiter=',')

        # cbiou = CBIOUTracker()
        tracker = CBIOUTracker({
            'buffer_scale1': 0.3,
            'buffer_scale2': 0.5,
            'buffer_scale3': 0.5,
            'match_high_score_dets_with_confirmed_trks_threshold' : 0.2,
            'match_low_score_dets_with_confirmed_trks_threshold' : 0.5,
            'match_remained_high_score_dets_with_unconfirmed_trks_threshold' : 0.3,
        })

        for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
            # gt_dets, dets = gt_dets_file[gt_dets_file[:,0] == frame_number][:, 1:6], detections[int(frame_number)][:, :5]
            dets = detections[detections[:, 0] == frame_number][:, 1:]
            tracker.update(dets)
            for track in Track.get_tracks([STATE_TRACKING]):
                file.write(f'{track.mot_format}\n')
            # if frame_number == 2:
            #     break
        file.close()
    seqmap.close()



if __name__ == '__main__':
    print('tracking...')
    run()
    print('evaluating...')
    evaluate()