from functools import partial

from langgraph.graph import END, StateGraph

from .clients.lm_studio import LMStudioClient
from .clients.minio_client import MinIOClient
from .config import AppConfig
from .nodes import assembler, check_cache, extractor, load_transcript, normalizer, translator, uploader, validator
from .nodes import title as title_node
from .nodes.exporters import ass, srt
from .nodes.transcriber import transcribe_chunks
from .state import CaptionState


def _make_minio(app_config: AppConfig) -> MinIOClient:
    minio = MinIOClient(
        endpoint=app_config.minio.endpoint,
        access_key=app_config.minio.access_key,
        secret_key=app_config.minio.secret_key,
        secure=app_config.minio.secure,
    )
    minio.ensure_bucket(app_config.minio.input_bucket)
    minio.ensure_bucket(app_config.minio.output_bucket)
    return minio


def build_transcription_graph(app_config: AppConfig):
    """Graph 1 — download, extract, chunk, transcribe, save JSON/CSV to disk."""
    minio = _make_minio(app_config)

    g = StateGraph(CaptionState)

    g.add_node("check_cache",       partial(check_cache, app_config=app_config))
    g.add_node("fetch_and_extract", partial(extractor.fetch_and_extract, app_config=app_config, minio=minio))
    g.add_node("transcribe_chunks", partial(transcribe_chunks, app_config=app_config))
    g.add_node("merge_and_save",    partial(assembler.merge_and_save, app_config=app_config))

    g.set_entry_point("check_cache")
    g.add_conditional_edges(
        "check_cache",
        lambda s: END if s.get("cache_hit") else "fetch_and_extract",
    )
    g.add_edge("fetch_and_extract", "transcribe_chunks")
    g.add_edge("transcribe_chunks", "merge_and_save")
    g.add_edge("merge_and_save",    END)

    return g.compile()


def build_translation_graph(app_config: AppConfig):
    """Graph 2 — load transcript from disk, translate, export, upload."""
    lm = LMStudioClient(
        base_url=app_config.lm_studio.base_url,
        model=app_config.lm_studio.model,
        api_key=app_config.lm_studio.api_key,
        timeout=app_config.lm_studio.timeout,
        max_retries=app_config.lm_studio.max_retries,
    )
    minio = _make_minio(app_config)

    g = StateGraph(CaptionState)

    g.add_node("load_transcript",    partial(load_transcript, app_config=app_config))
    g.add_node("normalize_segments", normalizer.normalize_segments)
    g.add_node("translate_title",    partial(title_node.translate_title, lm=lm))
    g.add_node("translate_segments", partial(translator.translate_segments, _app_config=app_config, lm=lm))
    g.add_node("validate_timeline",  validator.validate_timeline)
    g.add_node("export_srt",         partial(srt.export_srt, app_config=app_config))
    g.add_node("export_ass",         partial(ass.export_ass, app_config=app_config))
    g.add_node("upload_outputs",     partial(uploader.upload_outputs, app_config=app_config, minio=minio))

    g.set_entry_point("load_transcript")
    g.add_edge("load_transcript",    "normalize_segments")
    g.add_edge("normalize_segments", "translate_title")
    g.add_edge("translate_title",    "translate_segments")
    g.add_edge("translate_segments", "validate_timeline")
    g.add_edge("validate_timeline",  "export_srt")
    g.add_edge("export_srt",         "export_ass")
    g.add_edge("export_ass",         "upload_outputs")
    g.set_finish_point("upload_outputs")

    return g.compile()
