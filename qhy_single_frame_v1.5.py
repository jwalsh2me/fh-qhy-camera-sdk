#!/usr/bin/env python3
"""Capture one single frame from a QHY camera through the QHYCCD SDK.

Example:
    python qhy_single_frame.py 50
    python qhy_single_frame.py 20 --gain 80

The positional number is exposure time in milliseconds.
"""

import argparse
import ctypes as C
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from astropy.io import fits

LIBQHY = "/usr/local/lib/libqhyccd.so"
QHYCCD_SUCCESS = 0
QHYCCD_ERROR = 0xFFFFFFFF

# CONTROL_ID values used here (from QHYCCD SDK headers/manual)
CONTROL_GAIN = 6
CONTROL_OFFSET = 7
CONTROL_EXPOSURE = 8  # microseconds
CONTROL_USBTRAFFIC = 12  # lower = less USB bandwidth per transfer, more reliable


def load_sdk():
    if not Path(LIBQHY).exists():
        raise FileNotFoundError(
            f"Could not find {LIBQHY}. Install the QHYCCD ARM SDK first."
        )

    qhy = C.CDLL(LIBQHY)

    qhy.InitQHYCCDResource.argtypes = []
    qhy.InitQHYCCDResource.restype = C.c_uint32

    qhy.ReleaseQHYCCDResource.argtypes = []
    qhy.ReleaseQHYCCDResource.restype = C.c_uint32

    qhy.ScanQHYCCD.argtypes = []
    qhy.ScanQHYCCD.restype = C.c_uint32

    qhy.GetQHYCCDId.argtypes = [C.c_uint32, C.c_char_p]
    qhy.GetQHYCCDId.restype = C.c_uint32

    qhy.OpenQHYCCD.argtypes = [C.c_char_p]
    qhy.OpenQHYCCD.restype = C.c_void_p

    qhy.CloseQHYCCD.argtypes = [C.c_void_p]
    qhy.CloseQHYCCD.restype = C.c_uint32

    qhy.SetQHYCCDStreamMode.argtypes = [C.c_void_p, C.c_uint8]
    qhy.SetQHYCCDStreamMode.restype = C.c_uint32

    qhy.InitQHYCCD.argtypes = [C.c_void_p]
    qhy.InitQHYCCD.restype = C.c_uint32

    qhy.SetQHYCCDBitsMode.argtypes = [C.c_void_p, C.c_uint32]
    qhy.SetQHYCCDBitsMode.restype = C.c_uint32

    qhy.IsQHYCCDControlAvailable.argtypes = [C.c_void_p, C.c_int]
    qhy.IsQHYCCDControlAvailable.restype = C.c_uint32

    qhy.GetQHYCCDChipInfo.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_double), C.POINTER(C.c_double),
        C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
        C.POINTER(C.c_double), C.POINTER(C.c_double),
        C.POINTER(C.c_uint32),
    ]
    qhy.GetQHYCCDChipInfo.restype = C.c_uint32

    qhy.SetQHYCCDResolution.argtypes = [
        C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32
    ]
    qhy.SetQHYCCDResolution.restype = C.c_uint32

    qhy.SetQHYCCDBinMode.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32]
    qhy.SetQHYCCDBinMode.restype = C.c_uint32

    qhy.SetQHYCCDParam.argtypes = [C.c_void_p, C.c_int, C.c_double]
    qhy.SetQHYCCDParam.restype = C.c_uint32

    qhy.GetQHYCCDParam.argtypes = [C.c_void_p, C.c_int]
    qhy.GetQHYCCDParam.restype = C.c_double

    qhy.GetQHYCCDParamMinMaxStep.argtypes = [
        C.c_void_p, C.c_int,
        C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double)
    ]
    qhy.GetQHYCCDParamMinMaxStep.restype = C.c_uint32

    qhy.GetQHYCCDMemLength.argtypes = [C.c_void_p]
    qhy.GetQHYCCDMemLength.restype = C.c_uint32

    qhy.ExpQHYCCDSingleFrame.argtypes = [C.c_void_p]
    qhy.ExpQHYCCDSingleFrame.restype = C.c_uint32

    qhy.GetQHYCCDSingleFrame.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
        C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
        C.POINTER(C.c_uint8),
    ]
    qhy.GetQHYCCDSingleFrame.restype = C.c_uint32

    return qhy


def check(ret, message):
    if ret != QHYCCD_SUCCESS:
        raise RuntimeError(f"{message} (QHY return value {ret})")


def get_single_frame_retry(qhy, handle, width, height, out_bpp, channels, buffer,
                            exposure_ms, poll_interval_s=0.05, margin_s=2.0):
    """Call GetQHYCCDSingleFrame, retrying while the frame isn't ready yet.

    ExpQHYCCDSingleFrame does not reliably block until exposure+readout are
    done on all QHY SDK builds/cameras, so GetQHYCCDSingleFrame can return an
    error if called too soon. Retry with a short sleep until it succeeds or
    a generous timeout (exposure time + margin) elapses.
    """
    timeout_s = (exposure_ms / 1000.0) + margin_s
    deadline = perf_counter() + timeout_s
    last_ret = None

    while True:
        last_ret = qhy.GetQHYCCDSingleFrame(
            handle,
            C.byref(width), C.byref(height),
            C.byref(out_bpp), C.byref(channels), buffer,
        )
        if last_ret == QHYCCD_SUCCESS:
            return
        if perf_counter() >= deadline:
            raise RuntimeError(
                f"Could not retrieve image after retrying for {timeout_s:.1f}s "
                f"(last QHY return value {last_ret})"
            )
        sleep(poll_interval_s)


def get_param_range(qhy, handle, control_id):
    vmin = C.c_double()
    vmax = C.c_double()
    step = C.c_double()
    ret = qhy.GetQHYCCDParamMinMaxStep(
        handle, control_id, C.byref(vmin), C.byref(vmax), C.byref(step)
    )
    if ret == QHYCCD_SUCCESS:
        return vmin.value, vmax.value, step.value
    return None


def connect_camera_retry(qhy, timeout_s=8.0, poll_interval_s=0.5):
    """Scan for and open the first QHY camera, retrying while it's not yet
    visible on the bus.

    After a previous process closes the camera and releases the USB
    resource, the device can take a moment to re-enumerate. Calling
    ScanQHYCCD/OpenQHYCCD immediately afterward (e.g. back-to-back script
    invocations in a series) can otherwise fail with "No QHY camera found"
    or "SDK not support this camera now".
    """
    deadline = perf_counter() + timeout_s
    attempt = 0

    while True:
        attempt += 1
        ncam = qhy.ScanQHYCCD()
        if ncam != QHYCCD_ERROR and ncam >= 1:
            camera_id = C.create_string_buffer(128)
            if qhy.GetQHYCCDId(0, camera_id) == QHYCCD_SUCCESS:
                handle = qhy.OpenQHYCCD(camera_id)
                if handle:
                    print(f"Found {ncam} QHY camera(s) (attempt {attempt}).")
                    print("Camera:", camera_id.value.decode(errors="replace"))
                    return handle, camera_id

        if perf_counter() >= deadline:
            raise RuntimeError(
                f"No QHY camera found/openable after retrying for {timeout_s:.1f}s "
                f"({attempt} attempt(s)). Is the camera still connected and "
                f"fully released by any previous process?"
            )
        sleep(poll_interval_s)


def capture_at_exposure(qhy, handle, exposure_ms, count, current_gain, output_prefix):
    """Set exposure time on an already-open camera and capture `count` frames.

    The camera is not closed/reopened here -- callers should hold one open
    handle across every exposure value in a series to avoid USB
    re-enumeration between steps (see connect_camera_retry's docstring).
    """
    exposure_us = exposure_ms * 1000.0
    check(
        qhy.SetQHYCCDParam(handle, CONTROL_EXPOSURE, exposure_us),
        "Could not set exposure time",
    )
    actual_exposure_us = qhy.GetQHYCCDParam(handle, CONTROL_EXPOSURE)
    print(f"Exposure set to: {actual_exposure_us / 1000.0:.3f} ms")

    mem_length = qhy.GetQHYCCDMemLength(handle)
    if mem_length == 0:
        raise RuntimeError("Camera reported a zero-length image buffer")
    buffer = (C.c_uint8 * mem_length)()

    header = fits.Header()
    header["EXPTIME"] = (actual_exposure_us / 1_000_000.0, "Exposure time [s]")
    header["GAIN"] = (current_gain, "QHY SDK gain setting")

    series_start = perf_counter()

    for frame_num in range(1, count + 1):
        width = C.c_uint32()
        height = C.c_uint32()
        out_bpp = C.c_uint32()
        channels = C.c_uint32()

        t0 = perf_counter()
        # Blocking call: returns only once the exposure and readout finish.
        ret = qhy.ExpQHYCCDSingleFrame(handle)
        if ret == QHYCCD_ERROR:
            raise RuntimeError(
                f"Camera refused to start exposure {frame_num}/{count} "
                f"at {exposure_ms:g} ms"
            )

        get_single_frame_retry(
            qhy, handle, width, height, out_bpp, channels, buffer,
            exposure_ms=exposure_ms,
        )
        elapsed = perf_counter() - t0

        if channels.value != 1:
            raise RuntimeError(
                f"This simple test expected a mono frame but camera returned "
                f"{channels.value} channels."
            )

        bytes_per_pixel = out_bpp.value // 8
        nbytes = width.value * height.value * channels.value * bytes_per_pixel
        raw = np.ctypeslib.as_array(buffer)[:nbytes]

        if out_bpp.value == 16:
            image = raw.view("<u2").reshape(height.value, width.value).copy()
        elif out_bpp.value == 8:
            image = raw.reshape(height.value, width.value).copy()
        else:
            raise RuntimeError(f"Unexpected output bit depth: {out_bpp.value}")

        if count == 1:
            output = f"{output_prefix}.fits"
        else:
            output = f"{output_prefix}_{frame_num:02d}.fits"

        fits.writeto(output, image, header=header, overwrite=True)

        print(
            f"[{frame_num}/{count}] Saved: {output}  "
            f"{width.value}x{height.value}, {out_bpp.value}-bit, "
            f"min={image.min()}, max={image.max()}, mean={image.mean():.1f}, "
            f"{elapsed * 1000.0:.1f} ms"
        )

    series_elapsed = perf_counter() - series_start
    print(
        f"Captured {count} frame(s) at {exposure_ms:g} ms in {series_elapsed:.2f} s "
        f"({series_elapsed / count:.3f} s/frame average)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Capture full-frame 16-bit image(s) from a QHY camera."
    )
    parser.add_argument(
        "exposure_ms", type=float, nargs="?", default=None,
        help="Exposure time in milliseconds (ignored if --exposures is given)"
    )
    parser.add_argument(
        "--exposures", default=None,
        help="Comma-separated list of exposure times in ms to run in one "
             "camera session, e.g. '10,20,50,100,200,500,1000'. The camera "
             "is opened once and stays open across every value -- this "
             "avoids USB re-enumeration issues that can happen when closing "
             "and reopening the camera between separate process runs. "
             "Overrides the positional exposure_ms."
    )
    parser.add_argument(
        "--gain", type=float, default=None,
        help="Optional camera gain. If omitted, the SDK's current/default gain is used."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output FITS filename prefix (without .fits). Default: "
             "qhy_<exposure>ms. Ignored per-exposure when --exposures is "
             "given; each exposure value gets its own qhy_<exposure>ms prefix "
             "unless --output-dir is set."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write FITS files into when using --exposures "
             "(created if missing). Files are named exp_<ms>ms[_<nn>].fits."
    )
    parser.add_argument(
        "--usb-traffic", type=float, default=None,
        help="Optional CONTROL_USBTRAFFIC value. Lower = less USB bandwidth "
             "per transfer = more reliable on slow links/USB timeouts, at the "
             "cost of slower frame transfer. Try a low value (e.g. 0-10) if "
             "you see USB timeout errors. If omitted, SDK default is used."
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of frames to capture back-to-back at each exposure "
             "(camera stays open between frames; default: 1)"
    )
    parser.add_argument(
        "--connect-timeout", type=float, default=8.0,
        help="Seconds to retry detecting/opening the camera before giving up. "
             "Default: 8.0"
    )
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1.")

    if args.exposures:
        exposures_ms = [float(x) for x in args.exposures.split(",") if x.strip()]
        if not exposures_ms:
            raise ValueError("--exposures did not contain any values.")
    elif args.exposure_ms is not None:
        exposures_ms = [args.exposure_ms]
    else:
        raise ValueError("Provide either exposure_ms or --exposures.")

    for exp in exposures_ms:
        if exp <= 0:
            raise ValueError(f"Exposure time must be greater than zero (got {exp}).")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    qhy = load_sdk()
    handle = None

    try:
        check(qhy.InitQHYCCDResource(), "Could not initialize QHYCCD SDK")

        handle, camera_id = connect_camera_retry(qhy, timeout_s=args.connect_timeout)

        # 0 = single-frame capture mode. QHY recommends setting this before InitQHYCCD.
        check(qhy.SetQHYCCDStreamMode(handle, 0), "Could not select single-frame mode")
        check(qhy.InitQHYCCD(handle), "Could not initialize camera")
        check(qhy.SetQHYCCDBitsMode(handle, 16), "Could not select 16-bit output")

        chip_w = C.c_double()
        chip_h = C.c_double()
        image_w = C.c_uint32()
        image_h = C.c_uint32()
        pixel_w = C.c_double()
        pixel_h = C.c_double()
        bpp = C.c_uint32()

        check(
            qhy.GetQHYCCDChipInfo(
                handle,
                C.byref(chip_w), C.byref(chip_h),
                C.byref(image_w), C.byref(image_h),
                C.byref(pixel_w), C.byref(pixel_h), C.byref(bpp),
            ),
            "Could not read camera information",
        )

        print(f"Sensor image size: {image_w.value} x {image_h.value}")
        print(f"Physical pixel size: {pixel_w.value:.3f} x {pixel_h.value:.3f} um")

        check(qhy.SetQHYCCDBinMode(handle, 1, 1), "Could not set 1x1 binning")
        check(
            qhy.SetQHYCCDResolution(handle, 0, 0, image_w.value, image_h.value),
            "Could not set full-frame ROI",
        )

        gain_range = get_param_range(qhy, handle, CONTROL_GAIN)
        current_gain = qhy.GetQHYCCDParam(handle, CONTROL_GAIN)
        if gain_range:
            print(
                "Gain range reported by camera: "
                f"{gain_range[0]:g} to {gain_range[1]:g}, step {gain_range[2]:g}"
            )
        print(f"Current gain: {current_gain:g}")

        if args.gain is not None:
            if gain_range and not (gain_range[0] <= args.gain <= gain_range[1]):
                raise ValueError(
                    f"Requested gain {args.gain} is outside the reported range "
                    f"{gain_range[0]} to {gain_range[1]}."
                )
            check(
                qhy.SetQHYCCDParam(handle, CONTROL_GAIN, args.gain),
                "Could not set gain",
            )
            current_gain = qhy.GetQHYCCDParam(handle, CONTROL_GAIN)
            print(f"Gain set to: {current_gain:g}")

        if args.usb_traffic is not None:
            if qhy.IsQHYCCDControlAvailable(handle, CONTROL_USBTRAFFIC) != QHYCCD_SUCCESS:
                raise RuntimeError("This camera does not support CONTROL_USBTRAFFIC")
            traffic_range = get_param_range(qhy, handle, CONTROL_USBTRAFFIC)
            if traffic_range and not (traffic_range[0] <= args.usb_traffic <= traffic_range[1]):
                raise ValueError(
                    f"Requested USB traffic {args.usb_traffic} is outside the "
                    f"reported range {traffic_range[0]} to {traffic_range[1]}."
                )
            check(
                qhy.SetQHYCCDParam(handle, CONTROL_USBTRAFFIC, args.usb_traffic),
                "Could not set USB traffic",
            )
            print(f"USB traffic set to: {args.usb_traffic:g}")

        multi = len(exposures_ms) > 1
        for exp in exposures_ms:
            if multi or args.output_dir:
                base_dir = args.output_dir or "."
                output_prefix = str(Path(base_dir) / f"exp_{exp:g}ms")
            elif args.output:
                # Strip a .fits suffix if given; capture_at_exposure adds it.
                output_prefix = str(Path(args.output).with_suffix(""))
            else:
                output_prefix = f"qhy_{exp:g}ms"

            capture_at_exposure(
                qhy, handle, exp, args.count, current_gain, output_prefix
            )

    finally:
        if handle:
            qhy.CloseQHYCCD(handle)
        qhy.ReleaseQHYCCDResource()


if __name__ == "__main__":
    main()
