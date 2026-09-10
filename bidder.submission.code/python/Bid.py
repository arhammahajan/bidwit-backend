"""
Real-Time Bidding (RTB) Algorithm Implementation
DTU Hackathon - Demand Side Platform (DSP) Bidding Strategy

This module implements the Bidder interface for real-time bid optimization
combining LightGBM CTR (Click-Through Rate) and CVR (Conversion Rate) predictions.
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Suppress sklearn/lightgbm deprecation & version mismatch warnings
warnings.filterwarnings("ignore")

# Ensure the directory containing Bid.py is on sys.path for internal imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .BidRequest import BidRequest
    from .Bidder import Bidder
except ImportError:
    from BidRequest import BidRequest
    from Bidder import Bidder


# ==============================================================================
# Feature Extraction Helpers
# ==============================================================================

def xtract_device_family(user_agent_str: Optional[str]) -> str:
    """Parses user agent string and extracts device family."""
    if isinstance(user_agent_str, str):
        user_agent_lower = user_agent_str.lower()
        if (
            "mobile" in user_agent_lower
            or "android" in user_agent_lower
            or "iphone" in user_agent_lower
            or "ipad" in user_agent_lower
        ):
            if "ipad" in user_agent_lower:
                return "Tablet"
            return "Mobile"
        elif "tablet" in user_agent_lower:
            return "Tablet"
        elif (
            "windows" in user_agent_lower
            or "macintosh" in user_agent_lower
            or "linux" in user_agent_lower
        ):
            return "Desktop"
        else:
            return "Other"
    return "Unknown"


def xtract_os_family(user_agent_str: Optional[str]) -> str:
    """Parses user agent string and extracts OS family."""
    if isinstance(user_agent_str, str):
        user_agent_lower = user_agent_str.lower()
        if "windows" in user_agent_lower:
            return "Windows"
        elif "macintosh" in user_agent_lower or "macos" in user_agent_lower:
            return "MacOS"
        elif "android" in user_agent_lower:
            return "Android"
        elif (
            "ios" in user_agent_lower
            or "iphone" in user_agent_lower
            or "ipad" in user_agent_lower
        ):
            return "iOS"
        elif "linux" in user_agent_lower:
            return "Linux"
        else:
            return "Other"
    return "Unknown"


def xtract_browser_family(user_agent_str: Optional[str]) -> str:
    """Parses user agent string and extracts browser family."""
    if isinstance(user_agent_str, str):
        user_agent_lower = user_agent_str.lower()
        if "chrome" in user_agent_lower:
            return "Chrome"
        elif "firefox" in user_agent_lower:
            return "Firefox"
        elif "safari" in user_agent_lower and "chrome" not in user_agent_lower:
            return "Safari"
        elif "edge" in user_agent_lower:
            return "Edge"
        elif "msie" in user_agent_lower or "trident" in user_agent_lower:
            return "IE"
        else:
            return "Other"
    return "Unknown"


def xtract_network_class(network_class_str: Optional[str]) -> int:
    """Extracts IPv4 network class from IP address string."""
    if isinstance(network_class_str, str):
        try:
            first_octet = int(network_class_str.split(".")[0])
            if 1 <= first_octet <= 126:
                return 1
            elif 128 <= first_octet <= 191:
                return 2
            elif 192 <= first_octet <= 223:
                return 3
            elif 224 <= first_octet <= 239:
                return 4
            elif 242 <= first_octet <= 255:
                return 5
        except (ValueError, IndexError):
            return 0
    return 0


def _safe_int(val, default: int = 0) -> int:
    """Safely converts a value to integer, handling None, 'null', and NaNs."""
    if val is None or val == "" or str(val).strip().lower() == "null":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _cat_code(val) -> int:
    """
    Encodes categorical features. Matches single-row pandas categorical behavior
    used during model training: 0 for valid string, -1 for null/missing.
    """
    if val is None or val == "" or str(val).strip().lower() == "null" or (isinstance(val, float) and np.isnan(val)):
        return -1
    return 0


def _parse_timestamp(timestamp_val) -> Tuple[int, float]:
    """
    Safely extracts weekday (0-6) and float timestamp.
    Handles timestamps in 'yyyyMMddHHmmssSSS' or similar formats.
    """
    if timestamp_val is None or timestamp_val == "" or str(timestamp_val).strip().lower() == "null":
        return 0, 0.0

    ts_str = str(timestamp_val).strip()
    weekday = 0
    if len(ts_str) >= 8:
        try:
            year = int(ts_str[:4])
            month = int(ts_str[4:6])
            day = int(ts_str[6:8])
            weekday = datetime(year, month, day).weekday()
        except Exception:
            weekday = 0

    try:
        ts_float = float(ts_str)
    except Exception:
        ts_float = 0.0

    return weekday, ts_float


# ==============================================================================
# Main Bidding Class
# ==============================================================================

class Bid(Bidder):
    """
    Demand Side Platform (DSP) Bidding Engine.

    Implements getBidPrice(bidRequest) adhering to:
    - Execution Time: <= 5 milliseconds per request.
    - Memory Usage: <= 512 MB.
    """

    def __init__(self, model_dir: Optional[str] = None):
        """Initializes the bidder parameters and loads pre-trained models."""
        self.bidRatio = 90  # Percentage of requests to bid on (0-100)
        self.baseBidPrice = 50  # Base bid price in CPM

        base_path = Path(model_dir) if model_dir else CURRENT_DIR

        ctr_model_path = base_path / "model_ctr.pkl"
        cvr_model_path = base_path / "model_cvr.pkl"
        scaler_ctr_path = base_path / "scaler_ctr.pkl"
        scaler_cvr_path = base_path / "scaler_cvr.pkl"

        for required_file, name in [
            (ctr_model_path, "CTR model"),
            (cvr_model_path, "CVR model"),
            (scaler_ctr_path, "CTR scaler"),
            (scaler_cvr_path, "CVR scaler"),
        ]:
            if not required_file.exists():
                raise FileNotFoundError(f"{name} not found at {required_file}")

        self.ctr_model = joblib.load(ctr_model_path)
        self.cvr_model = joblib.load(cvr_model_path)
        self.scaler_ctr = joblib.load(scaler_ctr_path)
        self.scaler_cvr = joblib.load(scaler_cvr_path)

        # Cache boosters and scaler parameters for high-throughput, sub-millisecond inference
        self.ctr_booster = getattr(self.ctr_model, "booster_", None)
        self.cvr_booster = getattr(self.cvr_model, "booster_", None)
        self.ctr_mean = np.asarray(getattr(self.scaler_ctr, "mean_", []), dtype=np.float64)
        self.ctr_scale = np.asarray(getattr(self.scaler_ctr, "scale_", []), dtype=np.float64)
        self.cvr_mean = np.asarray(getattr(self.scaler_cvr, "mean_", []), dtype=np.float64)
        self.cvr_scale = np.asarray(getattr(self.scaler_cvr, "scale_", []), dtype=np.float64)

        # Advertiser conversion multipliers N (Score = clicks + N * conversions)
        self.advertiser_n_values = {
            1458: 0,   # Local e-commerce
            3358: 2,   # Software
            3386: 0,   # Global e-commerce
            3427: 0,   # Oil
            3476: 10,  # Tire
        }

    def _preprocess_bid_request_ctr(self, bidRequest: BidRequest) -> pd.DataFrame:
        """Preprocesses a single bid request into a DataFrame for the CTR model."""
        weekday, ts_float = _parse_timestamp(bidRequest.getTimestamp())
        bid_request_data = {
            'ua_browser': _cat_code(xtract_browser_family(bidRequest.getUserAgent())),
            'ua_device': _cat_code(xtract_device_family(bidRequest.getUserAgent())),
            'ua_os': _cat_code(xtract_os_family(bidRequest.getUserAgent())),
            'weekday': weekday,
            'AdvertiserID': _safe_int(bidRequest.getAdvertiserId(), default=0),
            'Payingprice': 0,  # Not available in real-time bid request
            'Adslotfloorprice': _safe_int(bidRequest.getAdSlotFloorPrice(), default=0),
            'Adslotformat': _cat_code(bidRequest.getAdSlotFormat()),
            'Adslotheight': _safe_int(bidRequest.getAdSlotHeight(), default=0),
            'Adslotvisibility': _cat_code(bidRequest.getAdSlotVisibility()),
            'Adslotwidth': _safe_int(bidRequest.getAdSlotWidth(), default=0),
            'Timestamp': ts_float,
        }
        return pd.DataFrame([bid_request_data])

    def _preprocess_bid_request_cvr(self, bidRequest: BidRequest) -> pd.DataFrame:
        """Preprocesses a single bid request into a DataFrame for the CVR model."""
        weekday, ts_float = _parse_timestamp(bidRequest.getTimestamp())
        bid_request_data = {
            'clicked': 0,  # Not available in real-time bid request
            'ua_os': _cat_code(xtract_os_family(bidRequest.getUserAgent())),
            'ua_device': _cat_code(xtract_device_family(bidRequest.getUserAgent())),
            'weekday': weekday,
            'Timestamp': ts_float,
            'AdvertiserID': _safe_int(bidRequest.getAdvertiserId(), default=0),
            'Payingprice': 0,  # Not available in real-time bid request
            'Adexchange': _safe_int(bidRequest.getAdExchange(), default=0),
            'Biddingprice': 0,  # Not available in real-time bid request
            'Adslotformat': _cat_code(bidRequest.getAdSlotFormat()),
            'Adslotheight': _safe_int(bidRequest.getAdSlotHeight(), default=0),
            'Region': _safe_int(bidRequest.getRegion(), default=0),
        }
        return pd.DataFrame([bid_request_data])

    def getBidPrice(self, bidRequest: BidRequest) -> int:
        """
        Predicts CTR and CVR using trained LightGBM models and computes optimal bid price.
        Returns:
            int: The bid price in local currency / CPM, or -1 if no bid is placed.
        """
        bidPrice = -1

        # Apply stochastic bidding ratio filter
        if random.randint(0, 99) < self.bidRatio:
            # CTR Prediction
            ctr_df = self._preprocess_bid_request_ctr(bidRequest)
            if self.ctr_booster is not None and len(self.ctr_mean) == 12:
                ctr_arr = ctr_df.to_numpy(dtype=np.float64)
                ctr_scaled = (ctr_arr - self.ctr_mean) / self.ctr_scale
                ctr_prediction = float(self.ctr_booster.predict(ctr_scaled)[0])
            else:
                ctr_features_scaled = self.scaler_ctr.transform(ctr_df)
                ctr_prediction = float(self.ctr_model.predict_proba(ctr_features_scaled)[:, 1][0])

            # CVR Prediction
            cvr_df = self._preprocess_bid_request_cvr(bidRequest)
            if self.cvr_booster is not None and len(self.cvr_mean) == 12:
                cvr_arr = cvr_df.to_numpy(dtype=np.float64)
                cvr_scaled = (cvr_arr - self.cvr_mean) / self.cvr_scale
                cvr_prediction = float(self.cvr_booster.predict(cvr_scaled)[0])
            else:
                cvr_features_scaled = self.scaler_cvr.transform(cvr_df)
                cvr_prediction = float(self.cvr_model.predict_proba(cvr_features_scaled)[:, 1][0])

            # Retrieve campaign conversion multiplier N
            advertiser_id = _safe_int(bidRequest.getAdvertiserId(), default=0)
            n_value = self.advertiser_n_values.get(advertiser_id, 1)

            # Calculate bid price based on expected value
            estimated_value = ctr_prediction * (1 + n_value * cvr_prediction)
            bidPrice = int(self.baseBidPrice * estimated_value)

            # Ensure bid price meets floor price threshold
            floor_price = _safe_int(bidRequest.getAdSlotFloorPrice(), default=0)
            bidPrice = max(bidPrice, floor_price)

            # Bidding bounds checks
            if bidPrice <= 0:
                bidPrice = -1
            elif bidPrice > 300:
                bidPrice = 300

        return bidPrice

    def getBidRequest(self, bid_data, bidder_instance: Bid, output_file: Optional[str] = None) -> Tuple[float, int, int]:
        """
        Processes a single bid request record, measures execution time, and records results.
        """
        bid_request = BidRequest()

        bid_request.setBidId(str(bid_data[0]))
        timestamp_str = str(bid_data[1])
        bid_request.setTimestamp(timestamp_str)

        bid_request.setVisitorId(str(bid_data[2]) if str(bid_data[2]).strip().lower() != 'null' else None)
        bid_request.setUserAgent(str(bid_data[3]) if str(bid_data[3]).strip().lower() != 'null' else None)
        bid_request.setIpAddress(str(bid_data[4]) if str(bid_data[4]).strip().lower() != 'null' else None)
        bid_request.setRegion(str(bid_data[5]) if str(bid_data[5]).strip().lower() != 'null' else None)
        bid_request.setCity(str(bid_data[6]) if str(bid_data[6]).strip().lower() != 'null' else None)
        bid_request.setAdExchange(str(bid_data[7]) if str(bid_data[7]).strip().lower() != 'null' else None)
        bid_request.setDomain(str(bid_data[8]) if str(bid_data[8]).strip().lower() != 'null' else None)
        bid_request.setUrl(str(bid_data[9]) if str(bid_data[9]).strip().lower() != 'null' else None)
        bid_request.setAnonymousURLID(str(bid_data[10]) if str(bid_data[10]).strip().lower() != 'null' else None)
        bid_request.setAdSlotID(str(bid_data[11]) if str(bid_data[11]).strip().lower() != 'null' else None)
        bid_request.setAdSlotWidth(str(bid_data[12]) if str(bid_data[12]).strip().lower() != 'null' else None)
        bid_request.setAdSlotHeight(str(bid_data[13]) if str(bid_data[13]).strip().lower() != 'null' else None)
        bid_request.setAdSlotVisibility(str(bid_data[14]) if str(bid_data[14]).strip().lower() != 'null' else None)
        bid_request.setAdSlotFormat(str(bid_data[15]) if str(bid_data[15]).strip().lower() != 'null' else None)
        bid_request.setAdSlotFloorPrice(str(bid_data[16]) if str(bid_data[16]).strip().lower() != 'null' else None)
        bid_request.setCreativeID(str(bid_data[17]) if str(bid_data[17]).strip().lower() != 'null' else None)
        bid_request.setAdvertiserId(str(bid_data[18]) if str(bid_data[18]).strip().lower() != 'null' else None)
        if len(bid_data) > 19:
            bid_request.setUserTags(str(bid_data[19]) if str(bid_data[19]).strip().lower() != 'null' else None)

        # Measure execution time with high-resolution timer
        start_time = time.perf_counter()
        bid_price = bidder_instance.getBidPrice(bid_request)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

        if output_file:
            try:
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp_str, bid_data[0], bid_request.advertiserId, execution_time, bid_price])
            except Exception:
                pass

        return execution_time, 1 if bid_price != -1 else 0, 1 if bid_price == -1 else 0

    def test_bidding_framework(
        self,
        bid_dataframe: pd.DataFrame,
        no_of_rows: int = 1000,
        output_file: Optional[str] = None,
    ) -> float:
        """Tests the Bid class with bid requests and measures execution time."""
        bidder_instance = Bid()
        subset = bid_dataframe.head(no_of_rows)
        print(f"Running bidding framework on {len(subset)} bid requests...")

        results = subset.apply(
            lambda row: self.getBidRequest(row, bidder_instance, output_file=output_file),
            axis=1,
        )

        total_execution_time = sum(result[0] for result in results)
        bid_count = sum(result[1] for result in results)
        no_bid_count = sum(result[2] for result in results)
        total_processed = bid_count + no_bid_count
        avg_execution_time = total_execution_time / total_processed if total_processed > 0 else 0.0

        print("\n--- Testing Summary ---")
        print(f"Total Bid Requests Processed: {total_processed}")
        print(f"Number of Bids Placed:        {bid_count} ({(bid_count / total_processed * 100) if total_processed else 0:.1f}%)")
        print(f"Number of No Bids:            {no_bid_count} ({(no_bid_count / total_processed * 100) if total_processed else 0:.1f}%)")
        print(f"Average Execution Time:       {avg_execution_time:.4f} ms per request")
        print(f"Constraint Check (<= 5.0 ms): {'PASSED' if avg_execution_time <= 5.0 else 'FAILED'}")

        return avg_execution_time


# ==============================================================================
# CLI and Self-Test Runner
# ==============================================================================

def run_self_test():
    """Runs an internal validation self-test across synthetic requests."""
    print("=" * 60)
    print("Running Bid Engine Self-Test")
    print("=" * 60)

    bidder = Bid()
    sample_advertisers = ["1458", "3358", "3386", "3427", "3476"]
    latencies = []

    print("\n1. Testing individual advertiser campaigns:")
    for adv in sample_advertisers:
        req = BidRequest()
        req.setBidId(f"test_bid_{adv}")
        req.setTimestamp("20130607000103501")
        req.setUserAgent("Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0")
        req.setAdvertiserId(adv)
        req.setAdSlotFloorPrice("5")
        req.setAdSlotHeight("250")
        req.setAdSlotWidth("300")
        req.setAdSlotVisibility("SecondView")
        req.setAdSlotFormat("Fixed")
        req.setAdExchange("2")
        req.setRegion("15")

        t0 = time.perf_counter()
        price = bidder.getBidPrice(req)
        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000
        latencies.append(latency)
        print(f"  Advertiser {adv}: Bid Price = ${price}/CPM (Latency: {latency:.4f} ms)")

    print("\n2. Latency benchmark over 200 consecutive requests:")
    benchmark_req = BidRequest()
    benchmark_req.setBidId("benchmark_req")
    benchmark_req.setTimestamp("20130607000103501")
    benchmark_req.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    benchmark_req.setAdvertiserId("3358")
    benchmark_req.setAdSlotFloorPrice("5")
    benchmark_req.setAdSlotHeight("250")
    benchmark_req.setAdSlotWidth("300")
    benchmark_req.setAdSlotVisibility("FirstView")
    benchmark_req.setAdSlotFormat("Fixed")
    benchmark_req.setAdExchange("1")
    benchmark_req.setRegion("1")

    bench_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        bidder.getBidPrice(benchmark_req)
        t1 = time.perf_counter()
        bench_times.append((t1 - t0) * 1000)

    avg_bench = sum(bench_times) / len(bench_times)
    p95_bench = float(np.percentile(bench_times, 95))
    print(f"  Average Latency: {avg_bench:.4f} ms")
    print(f"  95th Percentile: {p95_bench:.4f} ms")
    print(f"  Status:          {'PASSED (<= 5 ms)' if p95_bench <= 5.0 else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time Bidding (RTB) algorithm test runner.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to TSV dataset file (e.g. dataset/bid.07.txt).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of bid requests to process from dataset (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output CSV file for bid evaluation logs.",
    )
    args = parser.parse_args()

    # Look for candidate dataset paths
    candidate_paths = []
    if args.dataset:
        candidate_paths.append(Path(args.dataset))
    else:
        candidate_paths.extend([
            Path("dataset/bid.07.txt"),
            CURRENT_DIR / "dataset" / "bid.07.txt",
            CURRENT_DIR.parent.parent / "dataset" / "bid.07.txt",
        ])

    target_dataset = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            target_dataset = p
            break

    if target_dataset:
        print(f"Loading dataset from: {target_dataset}")
        test_df = pd.read_csv(
            target_dataset,
            sep='\t',
            header=None,
            na_values=["null"],
            low_memory=False,
        )
        tester = Bid()
        tester.test_bidding_framework(test_df, no_of_rows=args.rows, output_file=args.output)
    else:
        if args.dataset:
            print(f"Warning: Specified dataset not found at '{args.dataset}'.")
        print("No dataset file provided or found. Running self-test suite...\n")
        run_self_test()
