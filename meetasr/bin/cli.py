"""CLI entrypoint: meetasr transcribe / meetasr server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def cmd_transcribe(args):
    """Run transcription on one or more audio files."""
    from meetasr.auto.auto_pipeline import AutoPipeline

    if args.config:
        pipeline = AutoPipeline.from_yaml(args.config)
    else:
        # Minimal config — ASR only, no LLM
        pipeline = AutoPipeline.from_config({
            "asr": {"model": args.model, "device": args.device, "hub": args.hub},
            "vad": {"model": "fsmn-vad", "hub": args.hub},
            "punc": {"model": "ct-punc", "hub": args.hub} if not args.no_punc else None,
        })

    for audio_path in args.audio:
        if not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}", file=sys.stderr)
            continue

        print(f"\nProcessing: {audio_path}", file=sys.stderr)
        result = pipeline.transcribe(audio_path, language=args.language)

        if args.output_format == "json":
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        elif args.output_format == "srt":
            srt_text = result.to_srt()
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(audio_path))[0]
                out_path = os.path.join(args.output_dir, f"{stem}.srt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(srt_text)
                print(f"Saved SRT: {out_path}", file=sys.stderr)
            else:
                print(srt_text)
        else:  # text (default)
            print(result.text)


def cmd_summarize(args):
    """Run full meeting summarization."""
    from meetasr.auto.auto_pipeline import AutoPipeline

    if not args.config:
        print(
            "[ERROR] --config is required for summarization (needs LLM config).",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline = AutoPipeline.from_yaml(args.config)

    for audio_path in args.audio:
        if not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}", file=sys.stderr)
            continue

        print(f"\nProcessing: {audio_path}", file=sys.stderr)
        report = pipeline.summarize_meeting(audio_path, language=args.language)

        if args.output_format == "markdown":
            output = report.to_markdown()
        else:
            output = report.to_json(indent=2)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            ext = "md" if args.output_format == "markdown" else "json"
            out_path = os.path.join(args.output_dir, f"{stem}_report.{ext}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Saved report: {out_path}", file=sys.stderr)
        else:
            print(output)


def cmd_server(args):
    """Start the FastAPI server."""
    import uvicorn
    if args.config:
        os.environ.setdefault("MEETASR_CONFIG", args.config)
    uvicorn.run(
        "meetasr.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="meetasr",
        description="MeetASR — Meeting Speech Recognition + LLM Summarization",
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- transcribe ----
    t = subparsers.add_parser("transcribe", help="Transcribe audio to text")
    t.add_argument("audio", nargs="+", help="Audio file path(s)")
    t.add_argument("--model", default="sensevoice-small", help="ASR model name")
    t.add_argument("--device", default="cpu", help="Device: cpu | cuda:0")
    t.add_argument("--hub", default="ms", help="Model hub: ms | hf")
    t.add_argument("--language", default="auto", help="Language: auto | vi | zh | en")
    t.add_argument("--config", help="Path to meeting_config.yaml")
    t.add_argument("--no-punc", action="store_true", help="Skip punctuation restoration")
    t.add_argument("-f", "--output-format", default="text", choices=["text", "json", "srt"])
    t.add_argument("-o", "--output-dir", help="Output directory for SRT/JSON files")
    t.set_defaults(func=cmd_transcribe)

    # ---- summarize ----
    s = subparsers.add_parser("summarize", help="Full meeting summarization with LLM")
    s.add_argument("audio", nargs="+", help="Audio file path(s)")
    s.add_argument("--config", required=True, help="Path to meeting_config.yaml (with llm section)")
    s.add_argument("--language", default="vi", help="Output language: vi | en")
    s.add_argument("-f", "--output-format", default="json", choices=["json", "markdown"])
    s.add_argument("-o", "--output-dir", help="Output directory for report files")
    s.set_defaults(func=cmd_summarize)

    # ---- server ----
    sv = subparsers.add_parser("server", help="Start REST API server")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", default=8000, type=int)
    sv.add_argument("--config", default="meeting_config.yaml")
    sv.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    sv.set_defaults(func=cmd_server)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    args.func(args)


if __name__ == "__main__":
    main()
