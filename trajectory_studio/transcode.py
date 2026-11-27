"""Make a browser-compatible copy of a recorded camera stream."""

import os
from fractions import Fraction
from pathlib import Path
import sys

import av


def main(source, target):
    if hasattr(os, "nice"):
        os.nice(10)
    target = Path(target)
    temporary = target.with_suffix(".encoding.mp4")
    with av.open(source) as input_file:
        source_stream = input_file.streams.video[0]
        rate = source_stream.average_rate or 30
        with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as output_file:
            stream = output_file.add_stream("libx264", rate=rate)
            stream.width = source_stream.width
            stream.height = source_stream.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "ultrafast", "crf": "24", "threads": "1"}
            for index, frame in enumerate(input_file.decode(source_stream)):
                frame.pts = index
                frame.time_base = Fraction(rate.denominator, rate.numerator)
                for packet in stream.encode(frame):
                    output_file.mux(packet)
            for packet in stream.encode():
                output_file.mux(packet)
    temporary.replace(target)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
