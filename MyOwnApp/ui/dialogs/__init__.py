"""ui.dialogs — standalone dialog windows."""

from .training_progress import TrainingProgressDialog
from .move_progress     import MoveProgressDialog
from .inbox_training    import InboxTrainingDialog

__all__ = [
    'TrainingProgressDialog',
    'MoveProgressDialog',
    'InboxTrainingDialog',
]
