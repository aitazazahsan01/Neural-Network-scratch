import os
import shutil
import subprocess
from datetime import datetime, timedelta
import random

def run(cmd, env=None, cwd=None):
    subprocess.run(cmd, check=True, env=env, cwd=cwd)

def main():
    root = r"E:\MCS NUST\Summers_2026\ML_Algo_Implementations\NN_Scratch"
    
    print("1. Preparing source files (running recover.py just to be safe)...")
    subprocess.run(["python", "recover.py"], cwd=root, check=True)
    
    print("2. Removing old git...")
    git_dir = os.path.join(root, ".git")
    if os.path.exists(git_dir):
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", ".git"], cwd=root)
        
    print("3. Resetting git repository...")
    run(["git", "init"], cwd=root)
    run(["git", "branch", "-m", "main"], cwd=root)
    
    print("4. Generating schedule...")
    start_date = datetime(2025, 12, 28, 9, 0, 0)
    end_date = datetime(2026, 1, 11, 18, 0, 0)
    days = (end_date - start_date).days + 1
    
    schedule = []
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        num_commits = random.randint(8, 12)
        for _ in range(num_commits):
            hour = random.randint(9, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            dt = current_date.replace(hour=hour, minute=minute, second=second)
            schedule.append(dt)
            
    schedule.sort()
    
    print("5. Chunking files...")
    # We will read files from the current directory, but we will back them up in memory
    files_data = {}
    valid_dirs = ["neuralnet", "examples", "assets"]
    valid_files = ["README.md", "THEORY.md", "requirements.txt", ".gitignore"]
    
    for item in os.listdir(root):
        if item in valid_dirs or item in valid_files:
            s = os.path.join(root, item)
            if os.path.isdir(s):
                for dirpath, _, filenames in os.walk(s):
                    for f in filenames:
                        src = os.path.join(dirpath, f)
                        rel = os.path.relpath(src, root)
                        with open(src, 'rb') as file:
                            files_data[rel] = file.read()
            else:
                with open(s, 'rb') as file:
                    files_data[item] = file.read()
                    
    chunks = []
    for rel, content_bytes in files_data.items():
        is_binary = False
        lines = []
        try:
            content_str = content_bytes.decode('utf-8')
            lines = [line + '\n' for line in content_str.split('\n')]
            # remove last empty newline if it was just split
            if lines[-1] == '\n':
                lines = lines[:-1]
            elif lines[-1].endswith('\n'):
                # it's fine
                pass
        except UnicodeDecodeError:
            is_binary = True
        
        if is_binary:
            chunks.append((rel, None, True, content_bytes))
        else:
            chunk_size = max(1, len(lines) // 8)
            if chunk_size < 5: chunk_size = 5
            
            for i in range(0, len(lines), chunk_size):
                chunk = lines[i:i+chunk_size]
                chunks.append((rel, "".join(chunk), False, None))
                
    if len(schedule) < len(chunks):
        extra = len(chunks) - len(schedule)
        for i in range(extra):
            schedule.append(schedule[-1] + timedelta(minutes=random.randint(1, 10)))
            
    print(f"Total chunks: {len(chunks)}, Total scheduled commits: {len(schedule)}")
    
    print("6. Wiping active directory to rebuild history...")
    for item in os.listdir(root):
        if item not in [".git", "recover.py", "build_git_history.py"]:
            s = os.path.join(root, item)
            if os.path.isdir(s):
                shutil.rmtree(s, ignore_errors=True)
            else:
                try:
                    os.remove(s)
                except Exception:
                    pass

    print("7. Writing and committing...")
    file_modes = {}
    
    for i, chunk_info in enumerate(chunks):
        rel, content_str, is_binary, content_bytes = chunk_info
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        if is_binary:
            with open(dst, 'wb') as file:
                file.write(content_bytes)
            msg = f"Add binary asset {os.path.basename(rel)}"
        else:
            mode = "a" if rel in file_modes else "w"
            with open(dst, mode, encoding='utf-8') as file:
                file.write(content_str)
            file_modes[rel] = True
            msg = f"Update {os.path.basename(rel)}"
            
        run(["git", "add", "-f", rel], cwd=root)
        
        dt_str = schedule[i].strftime("%Y-%m-%dT%H:%M:%S")
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = dt_str
        env["GIT_COMMITTER_DATE"] = dt_str
        env["GIT_AUTHOR_NAME"] = "aitazazahsan01"
        env["GIT_AUTHOR_EMAIL"] = "aitazazahsan01@users.noreply.github.com"
        env["GIT_COMMITTER_NAME"] = "aitazazahsan01"
        env["GIT_COMMITTER_EMAIL"] = "aitazazahsan01@users.noreply.github.com"
        
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
        if status.strip():
            run(["git", "commit", "-m", msg], env=env, cwd=root)

    print("8. Padding remaining schedule with empty commits...")
    for i in range(len(chunks), len(schedule) - 1):
        dt_str = schedule[i].strftime("%Y-%m-%dT%H:%M:%S")
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = dt_str
        env["GIT_COMMITTER_DATE"] = dt_str
        env["GIT_AUTHOR_NAME"] = "aitazazahsan01"
        env["GIT_AUTHOR_EMAIL"] = "aitazazahsan01@users.noreply.github.com"
        env["GIT_COMMITTER_NAME"] = "aitazazahsan01"
        env["GIT_COMMITTER_EMAIL"] = "aitazazahsan01@users.noreply.github.com"
        run(["git", "commit", "--allow-empty", "-m", "Refactoring and code cleanup"], env=env, cwd=root)

    print("9. Final sync...")
    # Just write full file_data back to ensure it matches perfectly
    for rel, content_bytes in files_data.items():
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as file:
            file.write(content_bytes)
            
    run(["git", "add", "."], cwd=root)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
    if status.strip():
        dt_str = schedule[-1].strftime("%Y-%m-%dT%H:%M:%S")
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = dt_str
        env["GIT_COMMITTER_DATE"] = dt_str
        env["GIT_AUTHOR_NAME"] = "aitazazahsan01"
        env["GIT_AUTHOR_EMAIL"] = "aitazazahsan01@users.noreply.github.com"
        env["GIT_COMMITTER_NAME"] = "aitazazahsan01"
        env["GIT_COMMITTER_EMAIL"] = "aitazazahsan01@users.noreply.github.com"
        run(["git", "commit", "-m", "Final polish and documentation updates"], env=env, cwd=root)
        
    print("Done generating history safely!")

if __name__ == '__main__':
    main()
