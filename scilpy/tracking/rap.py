import numpy as np

def rap_continue(tracker, line, prev_direction):
    if len(line)>3:
        v_out = line[-2] - line[-3]
        v_out = v_out / np.linalg.norm(v_out)
        pos = line[-2] + tracker.propagator.step_size * np.array(v_out)
        line[-1] = pos
        return line, v_out
    return line, prev_direction

