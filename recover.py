import json
import os

transcript_path = r"C:\Users\SiliCon\.gemini\antigravity-ide\brain\2ddddb0c-0293-4a1c-8e13-f23f8cf8fa93\.system_generated\logs\transcript_full.jsonl"
root_dir = r"E:\MCS NUST\Summers_2026\ML_Algo_Implementations\NN_Scratch"

files = {}

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'PLANNER_RESPONSE' and 'tool_calls' in data:
                for tool in data['tool_calls']:
                    name = tool.get('name')
                    args = tool.get('args', {})
                    if name == 'write_to_file':
                        target = args.get('TargetFile')
                        if target and target.startswith(root_dir) and not target.endswith("generate_history.py"):
                            files[target] = args.get('CodeContent', '')
                    elif name == 'replace_file_content':
                        target = args.get('TargetFile')
                        if target in files:
                            target_content = args.get('TargetContent')
                            replacement = args.get('ReplacementContent')
                            if target_content in files[target]:
                                files[target] = files[target].replace(target_content, replacement)
                            else:
                                print(f"Could not apply replacement in {target}")
        except Exception as e:
            pass

for target, content in files.items():
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        f.write(content)
        
print(f"Recovered {len(files)} files.")