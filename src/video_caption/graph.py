from functools import partial

from langgraph.graph import END, StateGraph

from .clients.lm_studio import LMStudioClient
from .config import AppConfig
from .nodes import assembler, check_cache, extractor, load_transcript, normalizer, translator, validator
from .nodes import title as title_node
from .nodes.exporters import ass, srt
from .nodes.transcriber import transcribe_chunks
from .state import CaptionState


def build_transcription_graph(app_config: AppConfig):
    """Graph 1 — extract audio from the local video, transcribe, save JSONL to the work dir."""
    g = StateGraph(CaptionState)

    g.add_node("check_cache",       check_cache)
    g.add_node("extract_audio",     extractor.extract_audio)
    g.add_node("transcribe_chunks", partial(transcribe_chunks, app_config=app_config))
    g.add_node("merge_and_save",    assembler.merge_and_save)

    g.set_entry_point("check_cache")
    g.add_conditional_edges(
        "check_cache",
        lambda s: END if s.get("cache_hit") else "extract_audio",
    )
    g.add_edge("extract_audio",     "transcribe_chunks")
    g.add_edge("transcribe_chunks", "merge_and_save")
    g.add_edge("merge_and_save",    END)

    return g.compile()


def build_translation_graph(app_config: AppConfig):
    """Graph 2 — load transcript from disk, translate, export captions next to the video."""
    lm = LMStudioClient(
        base_url=app_config.lm_studio.base_url,
        model=app_config.lm_studio.model,
        api_key=app_config.lm_studio.api_key,
        timeout=app_config.lm_studio.timeout,
        max_retries=app_config.lm_studio.max_retries,
    )

    g = StateGraph(CaptionState)

    g.add_node("load_transcript",    partial(load_transcript, app_config=app_config))
    g.add_node("normalize_segments", normalizer.normalize_segments)
    g.add_node("translate_title",    partial(title_node.translate_title, lm=lm))
    g.add_node("translate_segments", partial(translator.translate_segments, _app_config=app_config, lm=lm))
    g.add_node("validate_timeline",  validator.validate_timeline)
    g.add_node("export_srt",         srt.export_srt)
    g.add_node("export_ass",         ass.export_ass)

    g.set_entry_point("load_transcript")
    g.add_edge("load_transcript",    "normalize_segments")
    g.add_edge("normalize_segments", "translate_title")
    g.add_edge("translate_title",    "translate_segments")
    g.add_edge("translate_segments", "validate_timeline")
    g.add_edge("validate_timeline",  "export_srt")
    g.add_edge("export_srt",         "export_ass")
    g.set_finish_point("export_ass")

    return g.compile()
