# MagicQuill — edit with a few brush strokes

[![MagicQuill](https://getgoingfast.pro/assets/images/magicquill.png)](https://getgoingfast.pro/tools/magicquill/)

Draw where you want an image to change, describe the result, and generate your
edit. Color strokes, added edges, erased edges, and automatic prompt suggestions
remain available. Use the editor on the PC or control it from your phone.

**[Get the member quick installer at Get Going Fast](https://getgoingfast.pro/tools/magicquill/)**

The member download sets up the app and its models automatically. This source
repository includes the files for a manual installation; the quick installer
is supplied separately for members.

## Manual Windows installation

Use an NVIDIA GPU, a current driver, and enough disk space for approximately
34 GB of models plus Python, dependencies, and saved images.

1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/) and
   [Git](https://git-scm.com/downloads/win), then open Command Prompt.
2. Clone this repository and enter its folder:

   ```bat
   git clone https://github.com/gjnave/GGF-MagicQuill.git
   cd GGF-MagicQuill
   ```

3. Create the private Python environment. UV downloads Python 3.10 if needed:

   ```bat
   uv venv --managed-python --python 3.10 .venv
   ```

4. Install CUDA PyTorch, then the app dependencies and drawing component:

   ```bat
   uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
   uv pip install --python .venv\Scripts\python.exe --no-deps gradio_magicquill-0.0.1-py3-none-any.whl
   uv pip check --python .venv\Scripts\python.exe
   ```

5. Download the model files into this app and configure the local vision encoder:

   ```bat
   .venv\Scripts\python.exe download_models.py
   .venv\Scripts\python.exe verify_install.py .
   ```

6. Launch the app:

   ```bat
   .venv\Scripts\python.exe -X utf8 app.py
   ```

## Use it from your phone

In Settings choose **My Wi-Fi** or **Temporary phone link**, save, and restart
the app. Open the yellow link shown in Settings on the phone. The PC runs the
models and must remain on. A remote password is optional; the local PC page
stays login-free. Drawing controls adapt to a narrow screen.

Edits save automatically in `outputs`; the finished-image panel also provides
a download button. Ctrl+Enter starts an edit. Parameters remain available below
the editor. Settings includes Release models and Stop server.

## Sources and licenses

This continues the GGF Windows repair validated September 30, 2026:
MagicQuill `8958300027238b259cfcabbc8aa38793d7c4ac47`, LLaVA
`c121f0432da27facab705978f83c4ada465e46fd`, Python 3.10, Torch 2.7.1/CUDA 12.8,
Gradio 5.4.0, Transformers 4.37.2, and Diffusers 0.31.0. Pydantic 2.10.6 is
pinned for compatibility with this Gradio release.

Original [MagicQuill](https://github.com/magic-quill/MagicQuill) is by its
authors and carries the included CC BY-NC 4.0 license. LLaVA's Apache-2.0 license
is retained in `MagicQuill/LLaVA/LICENSE`. Models download from the publishers'
repositories under their respective terms; no weights are included here.
