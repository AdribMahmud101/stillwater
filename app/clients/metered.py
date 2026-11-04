"""Opt-in metered adapters. Configure conservative, verified rates before use.

Ledger amounts are estimates from reported usage, not provider invoices. The
caller must choose an upper bound covering every charge and independently
account for uploads, GPU time, taxes, pricing tiers, and provider-side errors.
"""
from app.clients.usage import baseten_usage_estimate, sam_video_estimate
from app.pipeline.budget_runner import run_budgeted


class MeteredBaseten:
    def __init__(self, client, ledger, *, day, upper_bound, input_per_million, output_per_million):
        self.client, self.ledger, self.day, self.upper_bound = client, ledger, day, upper_bound
        self.input_rate, self.output_rate = input_per_million, output_per_million
        self.model = client.model if hasattr(client, 'model') else None

    def chat(self, messages, **options):
        def call():
            response, cached = self.client.chat_with_meta(messages, **options)
            cost = baseten_usage_estimate(response, input_per_million=self.input_rate,
                                           output_per_million=self.output_rate, cached=cached)
            return {'response': response, 'cached': cached, 'cost_usd': cost}
        result = run_budgeted(self.ledger, self.day, self.upper_bound, call)
        return {**result['response'], '_stillwater': {'cached': result['cached'],
                'estimated_cost_usd': str(result['cost_usd'])}}


class MeteredSAM:
    def __init__(self, client, ledger, *, day, upper_bound, per_frame, minimum_frames):
        self.client, self.ledger, self.day, self.upper_bound = client, ledger, day, upper_bound
        self.per_frame, self.minimum_frames = per_frame, minimum_frames

    def upload_video(self, video):
        return self.client.upload_video(video)

    def delete_file(self, file_id):
        return self.client.delete_file(file_id)

    def segment_video_file_with_usage(self, file_id, prompt):
        def call():
            result = self.client.segment_video_file_with_meta(file_id, prompt)
            cost = sam_video_estimate(result, per_frame=self.per_frame,
                                      minimum_frames=self.minimum_frames, cached=result['cached'])
            return {'response': result, 'cost_usd': cost}
        result = run_budgeted(self.ledger, self.day, self.upper_bound, call)
        return {**result['response'], 'estimated_cost_usd': str(result['cost_usd'])}

    def segment_video_file(self, file_id, prompt):
        return self.segment_video_file_with_usage(file_id, prompt)['text']
