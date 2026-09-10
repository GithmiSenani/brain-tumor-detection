import os
import sys
import argparse
from huggingface_hub import HfApi, login

def deploy_backend(space_id="PramudithaN/brain-tumor-backend", token=None):
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Write Token (from https://huggingface.co/settings/tokens): ").strip()

    if not token:
        print("[!] No Hugging Face token provided. Aborting deployment.")
        return False

    login(token=token)
    api = HfApi(token=token)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(repo_dir, "backend")

    print(f"[*] Deploying backend from '{backend_dir}' to Hugging Face Space: {space_id}...")

    # Check target Space
    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        print(f"[+] Target Space '{space_id}' found.")
    except Exception as e:
        print(f"[!] Space check info: {e}")
        try:
            print(f"[*] Attempting to create or verify Space '{space_id}' with Docker SDK...")
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True
            )
            print(f"[+] Created/verified Space '{space_id}' successfully.")
        except Exception as ce:
            print(f"[!] Note: {ce}")

    # Upload the backend application to the root of the Space
    print(f"[*] Uploading backend files to Space: {space_id} ...")
    api.upload_folder(
        folder_path=backend_dir,
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy updated FastAPI backend to Hugging Face Space",
        ignore_patterns=["__pycache__/*", ".env", "*.pyc"]
    )

    # Upload the NeuroAI pipeline module to NeuroAI/ in the Space
    neuro_ai_dir = os.path.join(repo_dir, "NeuroAI")
    if os.path.exists(neuro_ai_dir):
        print(f"[*] Uploading NeuroAI pipeline to Space '{space_id}/NeuroAI'...")
        api.upload_folder(
            folder_path=neuro_ai_dir,
            path_in_repo="NeuroAI",
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy updated NeuroAI inference engine and configs",
            ignore_patterns=["*.pth", "*.pt", "*.zip", "__pycache__/*", "*.png", "output_finetune/*.pth"]
        )

    print(f"\n[+] Successfully deployed backend & NeuroAI pipeline to Hugging Face Space!")
    print(f"[+] Live Space URL: https://huggingface.co/spaces/{space_id}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Backend to Hugging Face Space")
    parser.add_argument("--space", type=str, default=None, help="Target Hugging Face Space ID (e.g. PramudithaN/brain-tumor-backend or GithmiSenani/brain-tumor-backend)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write permission)")
    args = parser.parse_args()

    default_space = "PramudithaN/brain-tumor-backend"
    target_space = args.space or os.environ.get("HF_SPACE_ID") or default_space
    deploy_backend(space_id=target_space, token=args.token)
