"""Backward-compatible entry routed through actual structured skill generation."""
import argparse,json
from natural_backend import submit
from robot_adapters import UnsupportedRequest
def main():
 p=argparse.ArgumentParser();p.add_argument('instruction');p.add_argument('--prepare-only',action='store_true');p.add_argument('--robot-id',default='g2crsb90d');a=p.parse_args()
 try:print(json.dumps(submit(a.instruction,robot_id=a.robot_id,execute=not a.prepare_only),ensure_ascii=False))
 except UnsupportedRequest as exc:print(json.dumps(exc.details,ensure_ascii=False));raise SystemExit(2)
if __name__=='__main__':main()
