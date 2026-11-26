"""Shared display palette, importable outside Blender."""
import numpy as np
def color(u):
    return np.array([np.interp(np.clip(u,0,1),[0,.5,1],v) for v in [(48,224,239),(114,181,91),(148,106,69)]])
