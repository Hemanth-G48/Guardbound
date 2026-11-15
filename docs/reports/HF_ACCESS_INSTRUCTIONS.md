# HuggingFace Llama Model Download Instructions

## Account Status
- Logged in as: **Bala07123**
- Token: `hf_XXXXX` (replace with your actual token)

## Issue
Access to `meta-llama/Llama-3.1-8B-Instruct` requires approval.
Even though account is logged in, access needs to be granted via web interface.

## Steps to Get Access

### 1. Visit the Model Page
Go to: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### 2. Accept the License
- Click **"Agree and access repository"** button
- This grants access for your account

### 3. Download the Model
Once access is granted, run:
```bash
hf download meta-llama/Llama-3.1-8B-Instruct
```

Or with explicit token:
```bash
hf download meta-llama/Llama-3.1-8B-Instruct --token hf_XXXXX
```

## Alternative: Use Ollama with Llama3

If HuggingFace access is problematic, we can use Ollama:
```bash
ollama pull llama3
```

Llama3 via Ollama is already available and working in our experiments.

## For Windows Symlink Warning

If you see symlink warnings on Windows, set:
```bash
set HF_HUB_DISABLE_SYMLINKS_WARNING=TRUE
```

## Download Location
Default: `C:\Users\CSLAB\.cache\huggingface\hub\`
