import math
import unittest
from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth=[dict(video_id=str(i),label=str(y),provided_time_sec='10' if y else '') for i,y in enumerate([1,0,0,1,1])]
        self.pred=[dict(video_id=str(i),prediction=str(y),processing_time_sec=str(i+1),event_time_sec='12') for i,y in enumerate([1,1,0,0])]

    def test_known_answer_and_missing(self):
        r=evaluate(self.truth,self.pred)
        self.assertEqual(r['confusion'],dict(TP=1,FP=1,TN=1,FN=1))
        self.assertEqual((r['precision'],r['recall'],r['f1']),(.5,.5,.5))
        self.assertEqual(r['unprocessed_count'],1)
        self.assertEqual(r['organizer_hint_time']['mae_sec'],2)
        self.assertEqual(len(r['processing_times']),4)

    def test_empty_predictions(self):
        r=evaluate(self.truth,[])
        self.assertEqual(r['unprocessed_count'],5)
        self.assertEqual(r['f1'],0)

    def test_unknown_duplicate_invalid(self):
        for predictions in (self.pred+self.pred[:1],[dict(video_id='unknown')],[dict(video_id='0',prediction='2',processing_time_sec='1')],
                            [dict(video_id='0',prediction='1',processing_time_sec='nan')]):
            with self.assertRaises(ValueError):
                evaluate(self.truth,predictions)

    def test_failed_is_unprocessed(self):
        self.pred[0]['status']='error'
        self.assertEqual(evaluate(self.truth,self.pred)['unprocessed_count'],2)

    def test_split_filter(self):
        r=evaluate(self.truth,self.pred,{'2'})
        self.assertEqual(r['confusion']['TN'],1)
        self.assertEqual(r['processed_videos'],1)

    def test_events_need_review(self):
        with self.assertRaises(ValueError):
            evaluate(self.truth,self.pred,events=[],predicted_events=[])

    def test_events_one_to_one(self):
        gt=[dict(video_id='a',label='1',reviewed_label='1',review_status='complete')]
        pred=[dict(video_id='a',prediction='1',processing_time_sec='2')]
        event=dict(video_id='a',start_sec='10',end_sec='20',event_type='action')
        guess=dict(event,start_sec='11',end_sec='21')
        r=evaluate(gt,pred,label_column='reviewed_label',events=[event],predicted_events=[guess,guess])['events']
        self.assertEqual((r['TP'],r['FP'],r['FN']),(1,1,0))
        self.assertEqual(r['start_mae_sec'],1)
        wrong=dict(guess,event_type='different')
        r=evaluate(gt,pred,label_column='reviewed_label',events=[event],predicted_events=[wrong])['events']
        self.assertEqual((r['TP'],r['FP'],r['FN']),(0,1,1))

    def test_maximum_matching(self):
        gt=[dict(video_id='a',reviewed_label='1',review_status='complete')]
        pred=[dict(video_id='a',prediction='1',processing_time_sec='1')]
        def e(s,t):
            return dict(video_id='a',start_sec=str(s),end_sec=str(t),event_type='action')
        r=evaluate(gt,pred,label_column='reviewed_label',events=[e(0,10),e(4,14)],predicted_events=[e(2,12),e(0,8)])
        self.assertEqual(r['events']['TP'],2)
