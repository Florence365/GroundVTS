# Dataset paths use placeholders — replace <...> with your actual paths before running.

import json
import os
import random
import numpy as np
from collections import defaultdict


def convert_qvhighlights_highlight_to_llamafactory_json(
    jsonl_path,
    video_dir,
    save_path,
    prompt_templates=None,
    clip_duration=2.0,
    seed: int = 42
):
    """
    Convert QVHighlights highlight detection data to LLaMA-Factory format JSON.

    Args:
        jsonl_path: Path to the original QVHighlights JSONL file.
        video_dir: Directory containing video files.
        save_path: Output JSON file path.
        prompt_templates: Optional list of instruction templates.
        clip_duration: Duration of each clip in seconds (default 2.0).
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed line: {line}")
                continue

            query = item["query"].strip()
            video_name = item["vid"]
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            duration = item.get("duration", 0)

            relevant_clip_ids = item.get("relevant_clip_ids", [])
            saliency_scores = item.get("saliency_scores", [])

            if not relevant_clip_ids or not saliency_scores:
                continue

            clip_times = [[clip_id * clip_duration, (clip_id + 1) * clip_duration] for clip_id in relevant_clip_ids]

            # Average saliency across annotators
            saliency_mean = [np.mean(scores) for scores in saliency_scores]

            grade_dict = defaultdict(list)
            for i, score in enumerate(saliency_mean):
                start, end = clip_times[i]
                if score >= 3.5:
                    grade = "very important"
                elif score >= 2.5:
                    grade = "important"
                else:
                    grade = "less important"
                grade_dict[grade].append(f"{start:.1f}s to {end:.1f}s")

            segments = []
            for grade in ["very important", "important", "less important"]:
                if grade_dict[grade]:
                    segs = ", ".join(grade_dict[grade])
                    segments.append(f"{grade} from {segs}")
            assistant_text = "The highlights are: " + "; ".join(segments)

            if prompt_templates:
                prompt_template = random.choice(prompt_templates)
                user_prompt = prompt_template.format(query=query)
            else:
                user_prompt = f"<video>Please highlight the most important parts for the following event: {query}"

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_text
                    }
                ],
                "videos": [video_path]
            }

            data.append(sample)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Conversion complete, {len(data)} samples saved to {save_path}")


if __name__ == "__main__":
    jsonl_path = "<path/to/highlight_train_release.jsonl>"
    video_dir = "<path/to/QVHighlights/videos>"
    save_path = "<path/to/output/FT_qvhighlights_hl_train.json>"

    prompt_templates = [
        "<video>Please highlight the most exciting parts related to: {query}",
        "<video>Find the most relevant or important moments for: {query}",
        "<video>Which moments in the video best reflect: {query}?",
        "<video>Highlight the key segments that correspond to: {query}",
        "<video>Show the most interesting clips about: {query}",
        "<video>What are the highlight moments for: {query}?",
        "<video>Mark the time intervals that are most significant for: {query}",
    ]

    convert_qvhighlights_highlight_to_llamafactory_json(
        jsonl_path, video_dir, save_path, prompt_templates
    )
