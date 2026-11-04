import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from app.clients.costs import CostLedger, BudgetExceeded
from app.clients.metered import MeteredBaseten, MeteredSAM
from app.pipeline.budget_runner import BudgetAccountingError

class Baseten:
    def __init__(self,cached=False): self.cached=cached; self.calls=0
    def chat_with_meta(self,messages,**options):
        self.calls+=1
        return {'choices':[{'message':{'content':'ok'}}], 'usage':{'prompt_tokens':100,'completion_tokens':20}}, self.cached

class SAM:
    def __init__(self,cached=False): self.cached=cached
    def segment_video_file_with_meta(self,file_id,prompt):
        return {'text':'raw','video_frames_processed':20,'cached':self.cached}

class MeteredTests(unittest.TestCase):
    def test_baseten_measured_and_cached(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'costs.sqlite',Decimal('1'))
            base=Baseten(); client=MeteredBaseten(base,ledger,day='2026-11-03',upper_bound=Decimal('.1'),input_per_million=Decimal('1'),output_per_million=Decimal('2'))
            result=client.chat([{'role':'user','content':'hi'}])
            self.assertEqual(result['_stillwater']['estimated_cost_usd'],'0.00014')
            base.cached=True
            result=client.chat([{'role':'user','content':'hi'}])
            self.assertEqual(result['_stillwater']['estimated_cost_usd'],'0')

    def test_sam_unknown_usage_keeps_reservation(self):
        class Unknown(SAM):
            def segment_video_file_with_meta(self,*args): return {'text':'x','cached':False,'video_frames_processed':None}
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'costs.sqlite',Decimal('1'))
            client=MeteredSAM(Unknown(),ledger,day='2026-11-03',upper_bound=Decimal('1'),per_frame=Decimal('.001'),minimum_frames=50)
            with self.assertRaises(BudgetAccountingError): client.segment_video_file_with_usage('file-1','person')
            with self.assertRaises(BudgetExceeded): ledger.reserve('2026-11-03',Decimal('.01'))

    def test_sam_usage(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'costs.sqlite',Decimal('1'))
            client=MeteredSAM(SAM(),ledger,day='2026-11-03',upper_bound=Decimal('.1'),per_frame=Decimal('.001'),minimum_frames=50)
            result=client.segment_video_file_with_usage('file-1','person')
            self.assertEqual(result['estimated_cost_usd'],'0.050')

    def test_witness_using_real_adapters_meters_once(self):
        import io
        import json
        from app.clients.baseten import BasetenClient
        from app.clients.cache import CallCache
        from app.pipeline.witness import describe_container
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self,*args): self.close()
        attrs={'holds_water':'yes','covered':'no','water_visible':'unsure','exposed_to_rain':'yes','orientation':'upright','object_label':'bucket','evidence':'Open bucket.'}
        requests=[]
        def opener(req,timeout):
            requests.append(1)
            return Response(json.dumps({'choices':[{'message':{'content':json.dumps(attrs)}}], 'usage':{'prompt_tokens':100,'completion_tokens':20}}).encode())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ledger=CostLedger(root/'costs.sqlite',Decimal('1'))
            base=BasetenClient('test-secret',CallCache(root/'cache'),opener=opener)
            client=MeteredBaseten(base,ledger,day='2026-11-03',upper_bound=Decimal('.1'),input_per_million=Decimal('1'),output_per_million=Decimal('2'))
            self.assertEqual(describe_container(client,b'image','image/jpeg'),attrs)
            self.assertEqual(describe_container(client,b'image','image/jpeg'),attrs)
            self.assertEqual(len(requests),1)
            with __import__('sqlite3').connect(root/'costs.sqlite') as db:
                values=[Decimal(row[0]) for row in db.execute('select usd from costs')]
            self.assertEqual(values,[Decimal('.00014'),Decimal('0')])
