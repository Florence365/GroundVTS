# Dataset paths use placeholders — replace <...> with your actual paths before running.

import json
import os
import random


def convert_qvhighlights_to_llamafactory_json(jsonl_path, video_dir, save_path,
                                              prompt_templates=None,
                                              seed: int = 42):
    """
    Convert QVHighlights JSONL data to LLaMA-Factory format JSON.

    Args:
        jsonl_path: Path to the original QVHighlights JSONL file.
        video_dir: Directory containing video files.
        save_path: Output JSON file path.
        prompt_templates: Optional list of prompt templates.
        seed: Random seed.
    """
    random.seed(seed)

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
            video_path = os.path.join("QVHighlights/videos", f"{video_name}.mp4")

            # Each video may have multiple relevant_windows
            for window in item.get("relevant_windows", []):
                if len(window) != 2:
                    continue
                start, end = window

                if prompt_templates:
                    prompt_template = random.choice(prompt_templates)
                    user_prompt = prompt_template.format(query=query)
                else:
                    user_prompt = f"<video>At what point in the video did the following events occur: {query}? Output the start and end timestamps."

                sample = {
                    "messages": [
                        {
                            "content": user_prompt,
                            "role": "user"
                        },
                        {
                            "content": f"from {start}s to {end}s",
                            "role": "assistant"
                        }
                    ],
                    "videos": [
                        video_path
                    ]
                }
                data.append(sample)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Conversion complete, {len(data)} samples saved to {save_path}")


if __name__ == "__main__":
    jsonl_path = "<path/to/highlight_train_release.jsonl>"
    video_dir = "<path/to/QVHighlights/videos>"
    save_path = "<path/to/output/FT_qvhighlights_train.json>"

    prompt_templates = [
        "<video>At what point in the video did the following events occur: {query}? Output the start and end timestamps.",
        "<video>What is the location of the moment: {query}?",
        "<video>Find when the following event happens in the video: {query} Give me the start and end times.",
        "<video>Please indicate the start and end timestamps for the event: {query}",
        "<video>Please predict start and end time of the following moment: {query}",
        "<video>During which time interval does this happen in the video: {query}?",
        "<video>Locate the moment in the video where this occurs: {query} Provide start and end times.",
        "<video>For the video, when does this event take place: {query}? Answer with start and end timestamps.",
        "<video>I want to know the start and end times of the following event in the video: {query}",
        "<video>Could you tell me from what time to what time this happens: {query}?",
        "<video>Can you tell me the time window of this event: {query}?",
        "<video>Please find the timestamps that mark the occurrence of this event: {query}",
        "<video>Identify the start and end of the following event in the video: {query}"
    ]

    convert_qvhighlights_to_llamafactory_json(jsonl_path, video_dir, save_path, prompt_templates=prompt_templates)
