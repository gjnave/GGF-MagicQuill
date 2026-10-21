"""MagicQuill with a responsive editor and the standard GGF phone settings."""
import argparse
import asyncio
from pathlib import Path
import os
import socket
import threading
import time

import gradio as gr
from gradio_magicquill import MagicQuill
from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from access import MODES, load, save, verify
from engine import ENGINE, APP, image_data, read_image

LOCAL = PHONE = PHONE_ERROR = ""
SETTINGS = load()
CSS = (APP / "ui.css").read_text() + (APP / "mobile.css").read_text()
JS = """() => {
 if(!window.ggfMagicQuillShortcut){
   window.ggfMagicQuillShortcut=true;
   document.addEventListener('keydown',e=>{
     if(e.ctrlKey && e.key==='Enter' && !e.repeat){
       const root=document.querySelector('#ggf-generate');
       const button=root?.matches('button')?root:root?.querySelector('button');
       if(button&&!button.disabled&&button.getClientRects().length){e.preventDefault();button.click();}
     }
   });
 }
}"""


def phone_status():
    local = f"[{LOCAL}]({LOCAL})" if LOCAL else "Starting…"
    remote = f"[{PHONE}]({PHONE})" if PHONE else (PHONE_ERROR or ("off" if SETTINGS["mode"] == "local" else "Starting phone link…"))
    return f"Local PC: {local} · Phone: {remote}"


def activity():
    import html
    stage = html.escape(ENGINE.stage)
    if ENGINE.busy:
        elapsed = int(time.monotonic() - ENGINE.started)
        bar = f'<progress value="{ENGINE.step}" max="{ENGINE.total}"></progress>' if ENGINE.total else "<progress></progress>"
        return f"{bar} {stage} · {elapsed // 60}:{elapsed % 60:02d}"
    return stage


def refresh():
    return activity(), gr.update(interactive=not ENGINE.busy,
        value="Working…" if ENGINE.busy else "Generate edit · Ctrl+Enter"), phone_status()


def generate(*args):
    result = {}
    def run():
        try:
            result["value"], result["path"] = ENGINE.generate(*args)
        except Exception as error:
            result["error"] = error
    if ENGINE.busy:
        raise gr.Error("A job is already running. Wait or stop it first.")
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    yield gr.skip(), None, activity(), gr.update(interactive=False, value="Working…")
    while thread.is_alive():
        yield gr.skip(), gr.skip(), activity(), gr.update(interactive=False, value="Working…")
        time.sleep(0.4)
    if "error" in result:
        yield gr.skip(), gr.skip(), activity(), gr.update(interactive=True, value="Generate edit · Ctrl+Enter")
        raise gr.Error(str(result["error"]))
    yield result["value"], result["path"], activity(), gr.update(interactive=True, value="Generate edit · Ctrl+Enter")


def stop():
    if ENGINE.busy:
        import comfy.model_management as mm
        mm.interrupt_current_processing(True)
        ENGINE.stage = "Stop requested. Waiting for the current model operation…"
    return activity()


def release():
    if not ENGINE.lock.acquire(blocking=False):
        return "Stop the current job first."
    try:
        ENGINE.release_models()
        return "Models released. The next edit loads them again."
    finally:
        ENGINE.lock.release()


def stop_server():
    if ENGINE.busy:
        return "Stop the current job first."
    threading.Timer(1.5, lambda: os._exit(0)).start()
    return "App stopped. Restart RUN.bat on the PC to use it again."


def build_ui():
    from MagicQuill import folder_paths
    checkpoints = folder_paths.get_filename_list("checkpoints")
    default = next((name for name in checkpoints if "realisticVision" in name), checkpoints[0] if checkpoints else None)
    theme = gr.themes.Base(primary_hue="amber", neutral_hue="slate").set(
        body_background_fill="#07101f", body_background_fill_dark="#07101f",
        body_text_color="#e8eef9", body_text_color_dark="#e8eef9",
        block_background_fill="#111c2f", block_background_fill_dark="#111c2f",
        block_label_text_color="#e8eef9", block_label_text_color_dark="#e8eef9",
        input_background_fill="#091425", input_background_fill_dark="#091425",
        button_primary_background_fill="#f5b942", button_primary_background_fill_dark="#f5b942",
        button_primary_text_color="#091425", button_primary_text_color_dark="#091425")
    with gr.Blocks(title="MagicQuill · Get Going Fast", css=CSS, js=JS, theme=theme) as ui:
        with gr.Column(elem_id="ggf-shell"):
            gr.HTML('<div id="ggf-hero"><div class="kicker">GET GOING FAST · LOCAL AI</div><h1>MagicQuill</h1>'
                '<p>Draw what you want to change. Edit on your PC or from your phone.</p><div class="links">'
                '<a href="https://getgoingfast.pro/tools/magicquill/">Get Going Fast</a>'
                '<a href="https://youtube.com/@TheAIHobbyGuy">The AI Hobby Guy</a></div></div>')
            with gr.Tabs():
                with gr.Tab("Edit"):
                    gr.Markdown("Upload a picture, draw your change, and check the prompt below the canvas. "
                                "Use the wand to turn automatic prompt suggestions on or off.")
                    editor = MagicQuill(theme="dark")
                    run = gr.Button("Generate edit · Ctrl+Enter", variant="primary", elem_id="ggf-generate")
                    cancel = gr.Button("Stop current edit")
                    state = gr.HTML(activity(), elem_id="ggf-status")
                    output = gr.Image(label="Finished image · download or save to your phone", interactive=False)
                    with gr.Accordion("Parameters", open=False):
                        model = gr.Dropdown(checkpoints, value=default, label="Base model")
                        resolution = gr.Slider(256, 2048, value=512, step=64, label="Image resolution · set before uploading")
                        negative = gr.Textbox(label="Negative prompt", value="")
                        fine = gr.Radio(["enable", "disable"], value="disable", label="Fine edges")
                        grow = gr.Slider(0, 100, value=15, step=1, label="Selection expansion")
                        edge = gr.Slider(0, 5, value=0.55, step=0.01, label="Edge strength")
                        color = gr.Slider(0, 5, value=0.55, step=0.01, label="Color strength")
                        strength = gr.Slider(0, 5, value=1, step=0.01, label="Inpaint strength")
                        seed = gr.Number(value=-1, precision=0, label="Seed · -1 for a new result")
                        steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                        cfg = gr.Slider(0, 100, value=5, step=0.1, label="Prompt guidance")
                        sampler = gr.Dropdown(["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2"], value="euler_ancestral", label="Sampler")
                        scheduler = gr.Dropdown(["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"], value="karras", label="Scheduler")
                with gr.Tab("Settings"):
                    gr.Markdown("### Use it from your phone\nChoose Wi-Fi for your network or a temporary link for access elsewhere. "
                                "Save and restart the app. The PC runs the models and must stay on.")
                    mode = gr.Dropdown(list(MODES), value=next(k for k,v in MODES.items() if v == SETTINGS["mode"]), label="Phone access")
                    user = gr.Textbox(label="Remote username · optional", value=SETTINGS["username"])
                    password = gr.Textbox(label="Remote password · optional", type="password")
                    save_settings = gr.Button("Save phone settings")
                    saved = gr.Markdown()
                    links = gr.Markdown(phone_status(), elem_id="ggf-phone-status")
                    free = gr.Button("Release models / free GPU memory")
                    free_note = gr.Markdown("Models are also released after each edit.")
                    shutdown = gr.Button("Stop server")
                    confirm = gr.Button("Confirm stop · restart this app on the local PC to use it again", visible=False)
                    shutdown_note = gr.Markdown()
            timer = gr.Timer(2)
        run.click(generate, [editor, model, negative, fine, grow, edge, color, strength, seed, steps, cfg, sampler, scheduler],
                  [editor, output, state, run], concurrency_limit=1, show_progress="minimal", api_name="generate_edit")
        cancel.click(stop, outputs=state, queue=False)
        resolution.change(lambda v: setattr(ENGINE, "resolution", int(v)), resolution, queue=False)
        save_settings.click(save, [mode, user, password], saved, queue=False)
        free.click(release, outputs=free_note, queue=False)
        shutdown.click(lambda: gr.update(visible=True), outputs=confirm, queue=False)
        confirm.click(stop_server, outputs=shutdown_note, queue=False)
        timer.tick(refresh, outputs=[state, run, links], queue=False)
    return ui


def add_routes(app, remote=False):
    async def authorize(request):
        if not remote or not SETTINGS["hash"]:
            return
        token = request.cookies.get(f"access-token-{app.cookie_id}") or request.cookies.get(f"access-token-unsecure-{app.cookie_id}")
        if not app.tokens.get(token):
            raise HTTPException(401, "Sign into this app first.")

    @app.post("/magic_quill/process_background_img")
    async def background(request: Request):
        await authorize(request)
        picture = read_image(await request.json()).convert("RGB")
        ratio = ENGINE.resolution / min(picture.size)
        picture = picture.resize((round(picture.width * ratio), round(picture.height * ratio)))
        return image_data(picture)

    @app.post("/magic_quill/guess_prompt")
    async def guess(request: Request):
        await authorize(request)
        try:
            return await run_in_threadpool(ENGINE.guess, await request.json())
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error


def launch(port, remote=False):
    ui = build_ui().queue(default_concurrency_limit=1)
    for candidate in range(port, port + 20):
        try:
            app, local, public = ui.launch(server_name="0.0.0.0" if remote and SETTINGS["mode"] == "lan" else "127.0.0.1",
                server_port=candidate, share=remote and SETTINGS["mode"] == "public",
                auth=(lambda u,p: verify(u,p,SETTINGS)) if remote and SETTINGS["hash"] else None,
                prevent_thread_lock=True, inbrowser=False, show_error=True,
                allowed_paths=[str(APP / "outputs")])
            add_routes(app, remote)
            return local, public
        except OSError as error:
            if "Cannot find empty port" not in str(error):
                raise
    raise RuntimeError("No free MagicQuill port found.")


def main():
    global LOCAL, PHONE, PHONE_ERROR
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=7883)
    args = parser.parse_args()
    os.chdir(APP)
    (APP / "outputs").mkdir(exist_ok=True)
    LOCAL, _ = launch(args.port)
    print("Local PC:", LOCAL, flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(LOCAL)
    if SETTINGS["mode"] != "local":
        try:
            remote, public = launch(args.port + 1000, remote=True)
            if public:
                PHONE = public
            elif SETTINGS["mode"] == "lan":
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
                    connection.connect(("8.8.8.8", 80))
                    PHONE = remote.replace("0.0.0.0", connection.getsockname()[0])
            else:
                PHONE_ERROR = "The link service did not return an address. Restart to retry."
        except Exception as error:
            PHONE_ERROR = str(error)
        print(phone_status(), flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
