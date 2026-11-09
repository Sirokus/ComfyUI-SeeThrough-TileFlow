"""The only place the pipeline's numeric ranges live.

ComfyUI-See-through itself refuses seeds above 2**32-1 and resolutions
outside 512-2048 (``nodes.py`` in that package).  The pipeline widgets used
to allow much wider values, so a workflow could look valid and then fail --
or be silently clamped -- inside the model.  Both the widget ranges and the
run-time checks import these constants, and they are plain ints so this
module stays importable without ComfyUI.
"""

RES_MIN = 512
RES_MAX = 2048
RES_STEP = 64

DEPTH_MIN = -1          # -1 means "follow the layer resolution"
DEPTH_MAX = 2048
DEPTH_STEP = 64

SEED_MIN = 0
SEED_MAX = 2 ** 32 - 1

# The depth pass can either get its own seed or follow the layer seed; -1 is
# the "follow" sentinel, so a workflow saved before the split keeps producing
# exactly the pixels it used to.
DEPTH_SEED_FOLLOW = -1

STEPS_MIN = 1
STEPS_MAX = 100


def _range_text(low, high, step=None):
    text = "%d-%d" % (low, high)
    if low < 0:
        text = "-1（跟随层分辨率）或 %d-%d" % (max(1, low), high)
    if step:
        text += "，步长 %d" % step
    return text


def check_resolution(value):
    """Return ``value`` as int, or raise ``ValueError`` in Chinese."""
    value = int(value)
    if not RES_MIN <= value <= RES_MAX:
        raise ValueError(
            "resolution=%d 超出上游允许范围（%s）。"
            "分辨率越低越省显存，越高细节越多。"
            % (value, _range_text(RES_MIN, RES_MAX, RES_STEP)))
    return value


def check_steps(value):
    value = int(value)
    if not STEPS_MIN <= value <= STEPS_MAX:
        raise ValueError(
            "num_inference_steps=%d 超出范围（%d-%d）。"
            % (value, STEPS_MIN, STEPS_MAX))
    return value


def check_depth_resolution(value):
    value = int(value)
    if value == -1:
        return value
    if not 1 <= value <= DEPTH_MAX:
        raise ValueError(
            "depth_resolution=%d 超出上游允许范围（%s）。"
            % (value, _range_text(DEPTH_MIN, DEPTH_MAX, DEPTH_STEP)))
    return value


def check_seed(value):
    value = int(value)
    if not SEED_MIN <= value <= SEED_MAX:
        raise ValueError(
            "random_seed=%d 超出上游允许范围（%d-%d）。"
            % (value, SEED_MIN, SEED_MAX))
    return value


def check_all(resolution, num_inference_steps, depth_resolution, random_seed):
    """Validate the four model-facing numbers; return them as ints."""
    return (
        check_resolution(resolution),
        check_steps(num_inference_steps),
        check_depth_resolution(depth_resolution),
        check_seed(random_seed),
    )


def check_depth_seed(value):
    """``-1`` (follow the layer seed) or a normal seed."""
    value = int(value)
    if value == DEPTH_SEED_FOLLOW:
        return value
    if not SEED_MIN <= value <= SEED_MAX:
        raise ValueError(
            "depth_random_seed=%d 超出范围（%d（跟随拆层种子）或 %d-%d）。"
            % (value, DEPTH_SEED_FOLLOW, SEED_MIN, SEED_MAX))
    return value


def resolve_depth_seed(depth_seed, layer_seed):
    """The number actually handed to the depth model."""
    depth_seed = int(depth_seed)
    return int(layer_seed) if depth_seed == DEPTH_SEED_FOLLOW else depth_seed
