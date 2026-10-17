"""Evaluate video verdicts and optional fully reviewed event intervals."""
import argparse
import csv
import json
import math
from pathlib import Path


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def binary(value):
    if str(value) not in ('0', '1'):
        raise ValueError(f'Expected binary label, got {value!r}')
    return int(value)


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Expected finite nonnegative number: {value}')
    return result


def scores(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.
    r = tp/(tp+fn) if tp+fn else 0.
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)


def evaluate(truth, predictions, ids=None, label_column='label', events=None, predicted_events=None, iou_threshold=.5):
    gt = {}
    for row in truth:
        key = row['video_id']
        if key in gt:
            raise ValueError(f'Duplicate truth video: {key}')
        gt[key] = row
    selected = set(gt) if ids is None else set(ids)
    if not selected or selected-set(gt):
        raise ValueError('Empty selection or unknown video IDs')
    if label_column == 'reviewed_label' and any(gt[k].get('review_status') != 'complete' for k in selected):
        raise ValueError('Reviewed verdicts require complete review for every selected video')
    pred = {}
    for row in predictions:
        key = row['video_id']
        if key not in gt or key in pred:
            raise ValueError(f'Unknown or duplicate prediction: {key}')
        pred[key] = row
    counts = dict(TP=0, FP=0, TN=0, FN=0)
    errors, times, unprocessed, point_errors = [], [], [], []
    for key in sorted(selected):
        y = binary(gt[key][label_column])
        row = pred.get(key)
        if row is None or row.get('status', 'ok') != 'ok':
            unprocessed.append(key)
            continue
        p = binary(row['prediction'])
        elapsed = number(row['processing_time_sec'])
        times.append(dict(video_id=key, processing_time_sec=elapsed))
        kind = ('TP' if p else 'FN') if y else ('FP' if p else 'TN')
        counts[kind] += 1
        if p != y:
            errors.append(dict(video_id=key, expected=y, prediction=p, error=kind, suspected_cause='', notes=''))
        if y and p and row.get('event_time_sec') not in (None, '') and gt[key].get('provided_time_sec') not in (None, ''):
            point_errors.append(abs(number(row['event_time_sec'])-number(gt[key]['provided_time_sec'])))
    result = dict(reference=label_column, total_videos=len(selected), processed_videos=len(times),
                  unprocessed_count=len(unprocessed), unprocessed_video_ids=unprocessed,
                  coverage=len(times)/len(selected), confusion=counts,
                  **scores(counts['TP'], counts['FP'], counts['FN']),
                  processing_times=times, error_registry=errors,
                  organizer_hint_time=dict(matched_video_count=len(point_errors), mae_sec=sum(point_errors)/len(point_errors) if point_errors else None,
                                           limitation='One organizer hint per video; not full event localization quality'))
    if events is not None:
        if label_column != 'reviewed_label' or any(gt[k].get('review_status')!='complete' for k in selected):
            raise ValueError('Event evaluation requires complete team annotations and reviewed_label')
        if not 0 < iou_threshold <= 1:
            raise ValueError('IoU threshold must be in (0,1]')
        def validate(rows):
            for row in rows:
                if row['video_id'] not in gt or not row['event_type']:
                    raise ValueError('Unknown event video or missing event type')
                if number(row['end_sec']) <= number(row['start_sec']):
                    raise ValueError('Event must have positive duration')
        validate(events)
        validate(predicted_events or [])
        tp=fp=fn=0
        by_type={}
        boundaries=[]
        for key in sorted(selected-set(unprocessed)):
            actual=[r for r in events if r['video_id']==key]
            guessed=[r for r in (predicted_events or []) if r['video_id']==key]
            if bool(actual) != bool(binary(gt[key]['reviewed_label'])):
                raise ValueError(f'Event/video ground truth mismatch: {key}')
            edges=[]
            for i,a in enumerate(actual):
                for j,b in enumerate(guessed):
                    intersection=max(0., min(float(a['end_sec']),float(b['end_sec']))-max(float(a['start_sec']),float(b['start_sec'])))
                    union=max(float(a['end_sec']),float(b['end_sec']))-min(float(a['start_sec']),float(b['start_sec']))
                    overlap=intersection/union
                    if a['event_type']==b['event_type'] and overlap>=iou_threshold:
                        edges.append((overlap,i,j))
            # Maximum-cardinality bipartite matching, neighbors ordered by IoU.
            adjacency={i: [j for _,ai,j in sorted(edges, reverse=True) if ai==i] for i in range(len(actual))}
            matched={}
            def augment(i, seen):
                for j in adjacency[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in matched or augment(matched[j],seen):
                        matched[j]=i
                        return True
                return False
            for i in adjacency:
                augment(i,set())
            tp+=len(matched); fp+=len(guessed)-len(matched); fn+=len(actual)-len(matched)
            for category in sorted({r['event_type'] for r in actual+guessed}):
                bucket=by_type.setdefault(category,dict(TP=0,FP=0,FN=0))
                hits=sum(actual[i]['event_type']==category for i in matched.values())
                bucket['TP']+=hits
                bucket['FP']+=sum(r['event_type']==category for r in guessed)-hits
                bucket['FN']+=sum(r['event_type']==category for r in actual)-hits
            for j,i in matched.items():
                boundaries.append((abs(float(actual[i]['start_sec'])-float(guessed[j]['start_sec'])),abs(float(actual[i]['end_sec'])-float(guessed[j]['end_sec']))))
        for bucket in by_type.values():
            bucket.update(scores(bucket['TP'],bucket['FP'],bucket['FN']))
        result['events']=dict(TP=tp, FP=fp, FN=fn, **scores(tp,fp,fn), per_type=by_type, matched_count=len(boundaries),
                              start_mae_sec=sum(x[0] for x in boundaries)/len(boundaries) if boundaries else None,
                              end_mae_sec=sum(x[1] for x in boundaries)/len(boundaries) if boundaries else None,
                              iou_threshold=iou_threshold)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--splits',required=True)
    parser.add_argument('--split',choices=['train','validation','test'],default='test')
    parser.add_argument('--label-column',choices=['label','reviewed_label'],default='label')
    parser.add_argument('--events')
    parser.add_argument('--predicted-events')
    parser.add_argument('--output',default='artifacts/evaluation.json')
    args=parser.parse_args()
    if bool(args.events)!=bool(args.predicted_events):
        parser.error('--events and --predicted-events must be supplied together')
    rows=read(args.splits)
    if len({r['video_id'] for r in rows})!=len(rows):
        parser.error('Duplicate split IDs')
    report=evaluate(read(args.annotations),read(args.predictions),[r['video_id'] for r in rows if r['split']==args.split],args.label_column,
                    read(args.events) if args.events else None,read(args.predicted_events) if args.predicted_events else None)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for suffix,field,columns in [('errors','error_registry',['video_id','expected','prediction','error','suspected_cause','notes']),
                                 ('times','processing_times',['video_id','processing_time_sec'])]:
        with output.with_name(output.stem+'_'+suffix+'.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(report[field])
    print(json.dumps({k:v for k,v in report.items() if k not in ('processing_times','error_registry')},ensure_ascii=False,indent=2))
