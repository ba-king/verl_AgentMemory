"""Use the frozen candidate path; inherit all original retrieval behavior."""
from ...travel_v0.structured_record import bank as original
from ...travel_v0.single_action.session import bindings
from .config import HERE,CANDIDATE,candidate_config,save


class CandidateHybridAdapter(original.FreshHybridAdapter):
    def __init__(self,cache_root):
        cfg=candidate_config()
        # Local construction binding only; no source or global persistent change.
        with bindings(original,FROZEN=CANDIDATE):
            super().__init__(cache_root)
        assert self.backend.config==cfg['retriever']
        assert self.backend.config['top_k']==5


def verify_runtime():
    retriever=CandidateHybridAdapter(HERE/'embedding_cache')
    save('preflight/retriever_runtime.json',dict(candidate_read_explicitly=str(CANDIDATE),
        actual_backend_config=retriever.backend.config,actual_embedding_identity=retriever.encoder.identity,
        schema_and_renderer_unchanged=True,embedding_statistics=retriever.encoder.stats,
        retrieval_calls=0,api_requests=0,
        note='Actual frozen backend instantiated with Top5. No Teacher query or natural retrieval claimed by this initialization check.'))


if __name__=='__main__':verify_runtime()
