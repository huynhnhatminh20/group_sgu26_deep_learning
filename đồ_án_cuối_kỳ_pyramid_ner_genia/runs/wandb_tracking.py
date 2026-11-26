"""Optional W&B logging for future runs; never prints credentials."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import secrets
import traceback


_active = ContextVar("genia_wandb_run", default=None)


def _api_key():
    """Local private key file first, then the Kaggle Secret fallback."""
    try:
        from .wandb_auth import read_api_key
    except ImportError:
        from wandb_auth import read_api_key
    key = read_api_key()
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    key = UserSecretsClient().get_secret("Wandb")
    if not key:
        raise RuntimeError("Kaggle Secret Wandb is empty")
    return key


def _write_state(path, state):
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def tracking_run(output, experiment_id, config):
    """Use WANDB_PROJECT plus WANDB_API_KEY/Kaggle Secret, or WANDB_MODE=offline."""
    project = os.getenv("WANDB_PROJECT")
    if not project:
        yield None
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    id_path = output / "wandb_run_id.txt"
    if id_path.exists():
        run_id = id_path.read_text(encoding="ascii").strip()
    else:
        run_id = secrets.token_hex(8)
        with id_path.open("x", encoding="ascii") as stream:
            stream.write(run_id + "\n")
    state_path = output / "tracking_state.json"
    state = {"experiment_id": experiment_id, "wandb_run_id": run_id,
             "status": "RUNNING", "stop_reason": None}
    _write_state(state_path, state)
    run = None
    token = None
    try:
        import wandb
        mode = os.getenv("WANDB_MODE", "online")
        if mode == "online" and not os.getenv("WANDB_API_KEY"):
            key = _api_key()
            if not key:
                raise RuntimeError(
                    "WANDB_API_KEY is not set and no local private key file was found"
                )
            os.environ["WANDB_API_KEY"] = key
        run = wandb.init(project=project, entity=os.getenv("WANDB_ENTITY"),
                         id=run_id, resume="allow", name=experiment_id,
                         config=config, mode=mode, dir=str(output),
                         tags=["genia-v1", "diagnostic" if os.getenv("WANDB_DIAGNOSTIC")
                               else "official-run"])
        token = _active.set((run, state, state_path))
        yield run
        if state["status"] == "RUNNING":
            state["status"] = "COMPLETED"
        run.summary["project_status"] = state["status"]
        run.summary["stop_reason"] = state["stop_reason"]
        run.finish(exit_code=0)
    except BaseException as exc:
        failure_status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        state.update(status=failure_status, failure_reason=str(exc),
                     traceback=traceback.format_exc())
        if run is not None:
            run.summary["project_status"] = failure_status
            run.summary["failure_reason"] = str(exc)
            run.summary["traceback"] = state["traceback"]
            try:
                run.finish(exit_code=1)
            except Exception:
                pass  # Preserve the original training exception and local failure record.
        raise
    finally:
        if token is not None:
            _active.reset(token)
        _write_state(state_path, state)


def log_epoch(epoch, train_loss, valid):
    active = _active.get()
    if active is not None:
        run = active[0]
        run.log({"epoch": epoch, "train/loss": train_loss,
                 "valid/precision": valid["precision"],
                 "valid/recall": valid["recall"],
                 "valid/f1": valid["f1"]}, step=epoch)
        if valid["f1"] > run.summary.get("best_valid_f1", -1.0):
            run.summary["best_valid_f1"] = valid["f1"]
            run.summary["best_epoch"] = epoch


def set_run_metadata(values):
    active = _active.get()
    if active is not None:
        for key, value in values.items():
            active[0].summary[key] = value


def reconcile(output, kaggle_status):
    """Mark an unfinished local run after Kaggle reports a terminal job state."""
    path = Path(output) / "tracking_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["status"] == "RUNNING" and kaggle_status in ("ERROR", "COMPLETE", "CANCELLED"):
        state["status"] = "INTERRUPTED"
        state["reconciled_from_kaggle_status"] = kaggle_status
        _write_state(path, state)
    return state


def set_stop_reason(reason):
    active = _active.get()
    if active is not None:
        run, state, state_path = active
        run.summary["stop_reason"] = reason
        state["stop_reason"] = reason
        state["status"] = ("EARLY_STOPPED" if reason == "EARLY_STOP"
                           else "MAX_EPOCH_REACHED" if reason == "MAX_EPOCH_REACHED"
                           else "COMPLETED")
        _write_state(state_path, state)
