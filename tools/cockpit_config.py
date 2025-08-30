def get_config():
    return {
        "prompt": get_widget("prompt_input").text(),
        "negative_prompt": get_widget("negative_input").text(),
        "filename": get_widget("filename_input").text(),
        "steps": int(get_widget("steps_slider").value()),
        "cfg_scale": float(get_widget("cfg_slider").value()),
        "seed": int(get_widget("seed_input").text()),
        "resolution": get_widget("resolution_dropdown").currentText(),
        "amount": int(get_widget("amount_spinner").value()),
        "lock_seed": get_widget("lock_seed_toggle").isChecked(),
        "use_previous": get_widget("use_previous_toggle").isChecked()
    }