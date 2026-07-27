from .edsr import EDSRLite
from .rrdb import RRDBNet, UNetDiscriminator
from .swinir import SwinIRLite


def build_model(name="edsr_lite", scale=4, **kw):
    if name == "edsr_lite":
        return EDSRLite(scale=scale, **kw)
    if name == "swinir_lite":
        return SwinIRLite(scale=scale, **kw)
    if name == "rrdb":
        return RRDBNet(scale=scale, **kw)
    raise ValueError(f"unknown model {name!r}")


__all__ = ["EDSRLite", "SwinIRLite", "RRDBNet", "UNetDiscriminator", "build_model"]
