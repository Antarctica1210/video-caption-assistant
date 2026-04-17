from functools import partial

from langgraph.graph import END, StateGraph

from .clients.lm_studio import LMStudioClient
from .clients.minio_client import MinIOClient
from .config import AppConfig
from .nodes import assembler, chunker, extractor, translator, uploader, validator
from .nodes import title as title_node
from .nodes.exporters import ass, srt
from .nodes.transcriber import transcribe_chunks
from .state import CaptionState


def build_graph(config: AppConfig):
    lm = LMStudioClient(
        base_url=config.lm_studio.base_url,
        model=config.lm_studio.model,
        timeout=config.lm_studio.timeout,
        max_retries=config.lm_studio.max_retries,
        
    )
    minio = MinIOClient(
        endpoint=config.minio.endpoint,
        access_key=config.minio.access_key,
        secret_key=config.minio.secret_key,
        secure=config.minio.secure,
    )
    minio.ensure_bucket(config.minio.input_bucket)
    minio.ensure_bucket(config.minio.output_bucket)

    g = StateGraph(CaptionState)

    g.add_node("fetch_and_extract", partial(extractor.fetch_and_extract, config=config, minio=minio))
    g.add_node("chunk_audio",       partial(chunker.chunk_audio, config=config))
    g.add_node("transcribe_chunks", partial(transcribe_chunks, config=config))
    g.add_node("merge_and_save",    partial(assembler.merge_and_save, config=config))
    g.add_node("translate_title",   partial(title_node.translate_title, lm=lm))
    g.add_node("translate_segments",partial(translator.translate_segments, config=config, lm=lm))
    g.add_node("validate_timeline", validator.validate_timeline)
    g.add_node("export_srt",        partial(srt.export_srt, config=config))
    g.add_node("export_ass",        partial(ass.export_ass, config=config))
    g.add_node("upload_outputs",    partial(uploader.upload_outputs, config=config, minio=minio))

    g.set_entry_point("fetch_and_extract")
    g.add_edge("fetch_and_extract",  "chunk_audio")
    g.add_edge("chunk_audio",        "transcribe_chunks")
    g.add_edge("transcribe_chunks",  "merge_and_save")
    g.add_edge("merge_and_save",     "translate_title")
    g.add_edge("translate_title",    "translate_segments")
    g.add_edge("translate_segments", "validate_timeline")
    g.add_edge("validate_timeline",  "export_srt")
    g.add_edge("export_srt",         "export_ass")
    g.add_edge("export_ass",         "upload_outputs")
    g.add_edge("upload_outputs",     END)

    return g.compile()
