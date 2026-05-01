"""MLMixin — ML control panel, training, sorting, and clustering UI.

Methods owned by this mixin
───────────────────────────
setup_ml_control_panel()        Build the Standard + Advanced panel rows.
toggle_standard_section()       Collapse / expand the Standard section.
toggle_advanced_section()       Collapse / expand the Advanced section.
apply_hyperparameters()         Read entry widgets → update config + save.
reset_hyperparameters()         Restore defaults → repopulate entry widgets.
on_model_change(event)          Combobox: update model_type in config.
on_mode_change(event)           Combobox: update supervised_mode in config.
train_from_folders()            Queue "train_from_folders" task.
train_from_tags()               Queue "train_from_database" task.
auto_sort_images()              Run inference + apply tags with progress dialog.
run_clustering()                Queue unsupervised clustering task.
predict_current_image_tags()    Predict tags for the selected image.
show_metrics()                  Open the metrics window.
_start_training_with_progress(task_type, **kwargs)
_make_progress_callback(dialog) Return the callback wired to a progress dialog.

Current location: file_explorer.py, class FileExplorer.
"""


class MLMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
