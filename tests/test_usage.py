import unittest
from decimal import Decimal
from app.clients.usage import baseten_usage_estimate, sam_video_estimate, UsageError

class UsageTests(unittest.TestCase):
    def test_baseten_counts_reasoning_tokens_once(self):
        result={'usage':{'prompt_tokens':100, 'completion_tokens':20, 'completion_tokens_details':{'reasoning_tokens':15}}}
        amount=baseten_usage_estimate(result, input_per_million=Decimal('1'),output_per_million=Decimal('2'),cached=False)
        self.assertEqual(amount,Decimal('0.00014'))
        self.assertEqual(baseten_usage_estimate(result,input_per_million=Decimal('1'),output_per_million=Decimal('2'),cached=True),Decimal('0'))

    def test_sam_frame_minimum_is_supplied_and_cached_zero(self):
        result={'video_frames_processed':10}
        self.assertEqual(sam_video_estimate(result,per_frame=Decimal('0.001'),minimum_frames=50,cached=False),Decimal('.05'))
        self.assertEqual(sam_video_estimate(result,per_frame=Decimal('0.001'),minimum_frames=50,cached=True),Decimal('0'))

    def test_unknown_usage_and_nonfinite_prices_fail_closed(self):
        for result, rate in (({},Decimal('1')),({'usage':{'prompt_tokens':-1,'completion_tokens':2}},Decimal('1')),({'usage':{'prompt_tokens':1,'completion_tokens':2}},Decimal('NaN'))):
            with self.assertRaises(UsageError): baseten_usage_estimate(result,input_per_million=rate,output_per_million=Decimal('1'),cached=False)
        with self.assertRaises(UsageError): sam_video_estimate({},per_frame=Decimal('1'),minimum_frames=0,cached=False)
