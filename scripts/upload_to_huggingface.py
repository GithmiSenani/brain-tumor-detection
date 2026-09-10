import os
import sys
from huggingface_hub import HfApi, login

def upload_models(repo_id="PramudithaN/brain-tumor-models", token=None):
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Write Token (from https://huggingface.co/settings/tokens): ").strip()

    if not token:
        print("[!] No Hugging Face token provided. Aborting.")
        return

    login(token=token)
    api = HfApi()

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NeuroAI")
    
    swin_path = os.path.join(base_dir, "output_finetune", "best_model.pth")
    if not os.path.exists(swin_path):
        swin_path = os.path.join(base_dir, "best_model.pth")

    files_to_upload = [
        ("best_model.pth", swin_path),
        ("yolo_best.pt", os.path.join(base_dir, "yolo_best.pt")),
        ("densenet121_glioma.pth", os.path.join(base_dir, "densenet121_glioma.pth")),
        ("densenet121_meningioma.pth", os.path.join(base_dir, "densenet121_meningioma.pth")),
        ("densenet121_pituitary.pth", os.path.join(base_dir, "densenet121_pituitary.pth")),
        ("densenet121_notumor.pth", os.path.join(base_dir, "densenet121_notumor.pth")),
    ]

    print(f"[*] Uploading updated model weights to Hugging Face repo: {repo_id} ...")
    for filename, local_path in files_to_upload:
        if os.path.exists(local_path):
            print(f"  -> Uploading {filename} ({os.path.getsize(local_path) / (1024*1024):.2f} MB)...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Update {filename} with latest model weights",
            )
            print(f"  [+] Uploaded {filename} successfully!")
        else:
            print(f"  [!] File not found: {local_path}")

    print("\n[+] All model weights uploaded to Hugging Face successfully!")

if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "PramudithaN/brain-tumor-models"
    upload_models(repo_id=target_repo)
