import os
import numpy as np
from cbiou import CBIOUTracker
from track import Track
from track_state import STATE_UNCONFIRMED, STATE_TRACKING, STATE_LOST, STATE_DELETED, TrackState
import trackeval
import threading
import json

def evaluate(trackers_to_eval):
    # trackers_to_eval = ['cbiou', 'bytetrack', 'bytetrack-self']
    # trackers_to_eval = 'cbiou'
    dataset = 'MOT17'

    eval_config = {'USE_PARALLEL': True,
                    'NUM_PARALLEL_CORES': 8,
                    'BREAK_ON_ERROR': True,
                    'RETURN_ON_ERROR': False,
                    'LOG_ON_ERROR': '../outputs/error_log.txt',

                    'PRINT_RESULTS': False,
                    'PRINT_ONLY_COMBINED': False,
                    'PRINT_CONFIG': False,
                    'TIME_PROGRESS': False,
                    'DISPLAY_LESS_PROGRESS': True,

                    'OUTPUT_SUMMARY': False,
                    'OUTPUT_EMPTY_CLASSES': False,
                    'OUTPUT_DETAILED': False,
                    'PLOT_CURVES': False}

    dataset_config = {'GT_FOLDER': '../../.Datasets/MOT17/train/',
                        'TRACKERS_FOLDER': 'ablation_outputs',
                        'OUTPUT_FOLDER': None,
                        'TRACKERS_TO_EVAL': trackers_to_eval,
                        'CLASSES_TO_EVAL': ['pedestrian'],
                        'BENCHMARK': dataset if 'MOT' in dataset else 'MOT17',
                        # 'SPLIT_TO_EVAL': 'val',
                        'INPUT_AS_ZIP': False,
                        'PRINT_CONFIG': False,
                        'DO_PREPROC': True,
                        'TRACKER_SUB_FOLDER': '',
                        'OUTPUT_SUB_FOLDER': '',
                        'TRACKER_DISPLAY_NAMES': None,
                        'SEQMAP_FOLDER': None,
                        'SEQMAP_FILE': './trackeval/seqmap/%s/custom.txt' % dataset.lower(),
                        # 'SEQMAP_FILE': './trackeval/seqmap/%s/val.txt' % dataset.lower(),
                        'SEQ_INFO': None,
                        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                        'SKIP_SPLIT_FOL': True}


    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    os.makedirs('ablation_results', exist_ok=True)

    for tracker_to_eval in trackers_to_eval:

        hota = np.mean(res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
        idf1 = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1'].item()
        mota = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA'].item()
        motp = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTP'].item()
        assa = np.mean(res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
        deta = np.mean(res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()
        idsw = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['IDSW'].item()
        tp = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['CLR_TP']
        fp = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['CLR_FP']
        fn = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['CLR_FN']
        mt = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MT'].item()
        ml = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['ML'].item()
        
        obj = {
            'MOTA': mota,
            'HOTA': hota,
            'IDF1': idf1,
            'IDSW': idsw
        }

        with open(f'ablation_results/{tracker_to_eval}.json', 'w') as f:
            json.dump(obj, f)

        # file = open(f'ablation_results/{tracker_to_eval}-results.txt', 'w')
        # file.write(f'MOTA:    {mota}\n')
        # file.write(f'MOTP:    {motp}\n')
        # file.write(f'TP:      {tp}\n')
        # file.write(f'FP:      {fp}\n')
        # file.write(f'FN:      {fn}\n')
        # file.write(f'IDSW:    {idsw}\n')
        # file.write(f'IDF1:    {idf1}\n')
        # file.write(f'MT:      {mt}\n')
        # file.write(f'ML:      {ml}\n')
        # file.write(f'HOTA:    {hota}\n')
        # file.write(f'ASSA:    {assa}\n')
        # file.write(f'DETA:    {deta}\n')
        # file.write('\n')
        # file.write('\n')
        # count = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Count']
        # for key in count.keys():
        #     file.write(f'{key}{" "*(11 - len(key))}{count[key]}\n')
        # file.write('\n')
        # file.write('\n')
        # identity = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']
        # for key in identity.keys():
        #     file.write(f'{key}{" "*(8 - len(key))}{identity[key].item()}\n')
        # file.close()

def run(p):
    print(f'running with params {p}')
    seqs = ['MOT17-13-FRCNN', ]
    # seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ]
    name = f'{p[0]}-{p[1]}-{p[2]}'

    os.makedirs(f'ablation_outputs/{name}', exist_ok=True)

    # seqmap = open('./trackeval/seqmap/mot17/custom.txt', 'w')
    # seqmap.write('name\n')
    for seq in seqs:
        
        # seqmap.write(f'{seq}\n')
        file = open(f'ablation_outputs/{name}/{seq}.txt', 'w')
        # detections = detections_file[seq]
        detections = np.loadtxt(f'detections/bytetrack_x_mot17/{seq}.txt', delimiter=',')
        # detections = np.loadtxt('detections/MOT17-04-bytetrack-mot17-x.txt', delimiter=',')
        gt_dets_file = np.loadtxt(f'../../.Datasets/MOT17/train/{seq}/gt/gt.txt', delimiter=',')

        # cbiou = CBIOUTracker()
        tracker = CBIOUTracker({
            'buffer_scale1': p[0],
            'buffer_scale2': p[1],
            'buffer_scale3': p[2],
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
    # seqmap.close()

    # evaluate(name)

def cbiou_track(params):
    for p in params:
        run(p)




if __name__ == '__main__':
    all_params = []
    all_names = []
    for p1 in [0.1, 0.2, 0.3, 0.4 ,0.5]:
        for p2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for p3 in [0.2, 0.3, 0.4, 0.5 ,0.6]:
                # run((p1, p2, p3))
                # all_params.append((p1, p2, p3))
                all_names.append(f'{p1}-{p2}-{p3}')
    evaluate(all_names)
    

    # threads = []
    # for i in range(3):
    #     params = all_params[i*3: (i+1)*3]
    #     t = threading.Thread(target=cbiou_track, args=(params, ))
    #     threads.append(t)
    #     t.start()
    
    # for t in threads:
    #     t.join()


# if __name__ == '__main__':
#     threads = []
