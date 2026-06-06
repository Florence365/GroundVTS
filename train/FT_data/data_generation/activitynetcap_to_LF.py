# Dataset paths use placeholders — replace <...> with your actual paths before running.

import json
import os
import random


def convert_activitynetcaptions_to_llamafactory_json(json_path, video_dir, save_path,
                                                     prompt_templates=None,
                                                     seed: int = 42):
    """
    Convert ActivityNet-Captions JSON data to LLaMA-Factory format JSON.

    Args:
        json_path: Path to the original activitynet_captions.json file.
        video_dir: Directory containing video files.
        save_path: Output JSON file path.
        prompt_templates: Optional list of prompt templates.
        seed: Random seed.
    """
    random.seed(seed)

    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        video_name = item["video"]
        video_path = os.path.join("ActivityNet-Captions/Activity_Videos", f"{video_name}")

        timestamps = item.get("timestamps", [])
        sentences = item.get("sentences", [])

        for sent, window in zip(sentences, timestamps):
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            start, end = window

            if prompt_templates:
                prompt_template = random.choice(prompt_templates)
                user_prompt = prompt_template.format(query=sent.strip())
            else:
                user_prompt = f"<video>At what point in the video did the following events occur: {sent.strip()}? Output the start and end timestamps."

            sample = {
                "messages": [
                    {
                        "content": user_prompt,
                        "role": "user"
                    },
                    {
                        "content": f"from {round(start, 2)}s to {round(end, 2)}s",
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
    json_path = "<path/to/activitynet_captions_train.json>"
    video_dir = "<path/to/ActivityNet-Captions/Activity_Videos>"
    save_path = "<path/to/output/FT_activitynetcaptions_train.json>"

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

    convert_activitynetcaptions_to_llamafactory_json(json_path, video_dir, save_path, prompt_templates=prompt_templates)
