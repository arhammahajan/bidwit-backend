# BidWit — Real-Time Bidding (RTB) Machine Learning Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](requirements.txt)
[![Performance](https://img.shields.io/badge/Latency-%3C0.5ms%20per%20request-brightgreen.svg)](bidder.submission.code/python/Bid.py)

BidWit is a high-performance Demand Side Platform (DSP) Real-Time Bidding engine developed for the DTU RTB Bidding Hackathon. It predicts Click-Through Rates (CTR) and Conversion Rates (CVR) in real time using gradient-boosted decision trees (LightGBM) to submit optimal CPM bids in second-price ad auctions under advertiser budget constraints.

---

## Table of Contents

- [Overview & Objective](#overview--objective)
- [Architecture & Bidding Strategy](#architecture--bidding-strategy)
- [Project Layout](#project-layout)
- [Prerequisites & Installation](#prerequisites--installation)
- [Run Instructions](#run-instructions)
  - [1. Running the Automated Self-Test](#1-running-the-automated-self-test)
  - [2. Running on Competition Dataset Logs](#2-running-on-competition-dataset-logs)
  - [3. Programmatic Usage in Python](#3-programmatic-usage-in-python)
- [Performance & Competition Constraints](#performance--competition-constraints)
- [Dataset Schema](#dataset-schema)
- [License & Credits](#license--credits)

---

## Overview & Objective

In digital advertising, Real-Time Bidding (RTB) allows advertisers to evaluate and bid on ad impressions sequentially within milliseconds.

### Optimization Goal
Maximize advertiser outcomes subject to a fixed budget:
$$\text{Score} = \text{Total Clicks} + N \times \text{Total Conversions}$$

The conversion weight $N$ reflects the relative importance of conversions for different advertiser verticals:

| Advertiser ID | $N$ Multiplier | Vertical / Industry Category |
| :---: | :---: | :--- |
| **1458** | 0 | Local e-commerce |
| **3358** | 2 | Software |
| **3386** | 0 | Global e-commerce |
| **3427** | 0 | Oil |
| **3476** | 10 | Tire |

### Auction Mechanics
The ad exchange operates a **second-price auction**:
- The highest bidder wins the ad impression.
- The winning bidder pays the second-highest submitted bid price (market price).
- If no bid is placed or the bid is less than or equal to 0, the bidder submits `-1`.

---

## Architecture & Bidding Strategy

```
                          ┌───────────────────────────┐
                          │ Incoming BidRequest Event │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [ CTR Preprocessor ]          [ CVR Preprocessor ]
                         │                             │
                         ▼                             ▼
                 (StandardScaler)              (StandardScaler)
                         │                             │
                         ▼                             ▼
                 { LightGBM CTR Model }        { LightGBM CVR Model }
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         Estimated Value Computation:
                 P(CTR) * (1 + N_advertiser * P(CVR))
                                        │
                                        ▼
                         Floor Price & Bid Bounds Check
                                        │
                                        ▼
                         Final CPM Bid Price (or -1)
```

1. **Feature Extraction**:
   - User-Agent parsing extracts device family (`Mobile`, `Tablet`, `Desktop`), OS family (`Windows`, `MacOS`, `Android`, `iOS`, `Linux`), and browser family (`Chrome`, `Firefox`, `Safari`, `Edge`, `IE`).
   - High-speed timestamp parsing extracts weekday index and float timestamp.
   - Ad slot characteristics: floor price, slot width, slot height, visibility, and format.
2. **Dual-Model Inference**:
   - **CTR Classifier**: LightGBM model estimating the probability of click $P(\text{click} \mid \mathbf{x})$.
   - **CVR Classifier**: LightGBM model estimating the probability of conversion $P(\text{conv} \mid \mathbf{x})$.
   - Models run using low-level booster evaluation to guarantee sub-millisecond per-request latency.
3. **Bidding Formula**:
   $$\text{Estimated Value} = P(\text{CTR}) \times (1 + N \times P(\text{CVR}))$$
   $$\text{Bid} = \max(\lfloor \text{BaseBid} \times \text{Estimated Value} \rfloor, \text{FloorPrice})$$
   Bids are capped at 300 CPM, and non-bids return `-1`.

---

## Project Layout

```
bidwit-backend/
├── LICENSE                                # MIT License
├── README.md                              # Project documentation & run guide
├── requirements.txt                       # Python dependencies
├── output.txt                             # Recorded benchmark execution logs
├── city.txt                               # City code to name mapping
├── region.txt                             # Region code to name mapping
├── user.profile.tags.txt                  # User demographic & interest tag dictionary
└── bidder.submission.code/
    └── python/
        ├── Bid.py                         # Main Bidding algorithm & test runner
        ├── Bidder.py                      # Abstract Bidder interface
        ├── BidRequest.py                  # BidRequest data model
        ├── model_ctr.pkl                  # Serialized LightGBM CTR model
        ├── model_cvr.pkl                  # Serialized LightGBM CVR model
        ├── scaler_ctr.pkl                 # StandardScaler for CTR features
        ├── scaler_cvr.pkl                 # StandardScaler for CVR features
        └── requirements.txt               # Module dependency specification
```

---

## Prerequisites & Installation

- **Python Version**: Python 3.9 (recommended by competition specification; compatible with 3.9 through 3.13).
- **Git**

### 1. Clone the repository
```bash
git clone https://github.com/arhammahajan/bidwit-backend.git
cd bidwit-backend
```

### 2. Set up virtual environment

Using Python `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, using [`uv`](https://github.com/astral-sh/uv):
```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Run Instructions

### 1. Running the Automated Self-Test
The engine includes a built-in test suite that executes out-of-the-box without requiring external dataset files. It validates model loading, tests each advertiser campaign, and runs a 200-request latency benchmark:

```bash
python bidder.submission.code/python/Bid.py
```

**Expected Output:**
```text
No dataset file provided or found. Running self-test suite...

============================================================
Running Bid Engine Self-Test
============================================================

1. Testing individual advertiser campaigns:
  Advertiser 1458: Bid Price = $-1/CPM (Latency: 0.0063 ms)
  Advertiser 3358: Bid Price = $5/CPM (Latency: 0.8148 ms)
  Advertiser 3386: Bid Price = $-1/CPM (Latency: 0.0019 ms)
  Advertiser 3427: Bid Price = $-1/CPM (Latency: 0.0009 ms)
  Advertiser 3476: Bid Price = $5/CPM (Latency: 0.8975 ms)

2. Latency benchmark over 200 consecutive requests:
  Average Latency: 0.2903 ms
  95th Percentile: 0.3270 ms
  Status:          PASSED (<= 5 ms)
============================================================
```

### 2. Running on Competition Dataset Logs
To evaluate on TSV bid request logs (such as `dataset/bid.07.txt`):

```bash
# Evaluate the first 1,000 requests (default)
python bidder.submission.code/python/Bid.py --dataset dataset/bid.07.txt

# Specify row count and log output path
python bidder.submission.code/python/Bid.py \
  --dataset dataset/bid.07.txt \
  --rows 5000 \
  --output evaluation_results.csv
```

**Options:**
- `--dataset <path>`: Path to tab-separated bid log file.
- `--rows <int>`: Number of requests to process (default: `1000`).
- `--output <path>`: Optional CSV file destination to log `[timestamp, bidId, advertiserId, execution_time_ms, bid_price]`.

### 3. Programmatic Usage in Python
To integrate `Bid` directly into custom simulation pipelines or test harnesses:

```python
import sys
from pathlib import Path

# Add python submission folder to path
sys.path.insert(0, "bidder.submission.code/python")

from Bid import Bid
from BidRequest import BidRequest

# Initialize bidder (loads models and scalers)
bidder = Bid()

# Construct a bid request
request = BidRequest()
request.setBidId("req_sample_001")
request.setTimestamp("20130607000103501")
request.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0")
request.setAdvertiserId("3358")
request.setAdSlotFloorPrice("5")
request.setAdSlotWidth("300")
request.setAdSlotHeight("250")
request.setAdSlotVisibility("FirstView")
request.setAdSlotFormat("Fixed")
request.setAdExchange("2")
request.setRegion("15")

# Get optimal bid price
bid_price = bidder.getBidPrice(request)
print(f"Optimal Bid Price: ${bid_price} CPM")  # e.g., 5 or -1
```

---

## Performance & Competition Constraints

| Constraint | Competition Limit | BidWit Actual | Status |
| :--- | :---: | :---: | :---: |
| **Execution Time per Request** | $\le 5.0\text{ ms}$ | **$\approx 0.28\text{ ms}$** ($\approx 17\times$ faster) | **PASSED** |
| **Memory Usage (RSS)** | $\le 512\text{ MB}$ | **$\approx 135\text{ MB}$** | **PASSED** |
| **Package / Submission Size** | $\le 100\text{ MB}$ | **$< 2\text{ MB}$** | **PASSED** |
| **Interface Compliance** | `Bid` implements `Bidder` | Fully compliant (`getBidPrice` returns `int`) | **PASSED** |
| **Cross-Platform Compatibility** | Windows / Linux / macOS | Dynamic relative paths, no hardcoded paths | **PASSED** |

---

## Dataset Schema

Bid log files are formatted as tab-separated values (TSV) with the following structure:

| Col # | Field | Description | Example |
| :---: | :--- | :--- | :--- |
| **0** | `BidID` | Unique bid request identifier (hashed) | `014600008...3f5a4f1121` |
| **1** | `Timestamp` | Format: `yyyyMMddHHmmssSSS` | `20130607000103501` |
| **2** | `VisitorID` | Internal user cookie identifier | `35605620124522340227135` |
| **3** | `User-Agent` | Device, OS, and browser string | `Mozilla/5.0 (Windows NT 6.1...)` |
| **4** | `IP` | Anonymized user IP address | `118.81.189.*` |
| **5** | `Region` | Geographic region code (see `region.txt`) | `15` |
| **6** | `City` | Geographic city code (see `city.txt`) | `16` |
| **7** | `Adexchange` | Ad exchange platform identifier | `2` |
| **8** | `Domain` | Hashed publisher domain | `e80f4ac7...c01ad1a049` |
| **9** | `URL` | Hashed publisher URL | `hz55b01000...3d6f274121` |
| **10** | `AnonymousURLID`| Masked URL identifier | `null` |
| **11** | `AdslotID` | Ad slot identifier | `21476898764813` |
| **12** | `Adslotwidth` | Width in pixels | `300` |
| **13** | `Adslotheight` | Height in pixels | `250` |
| **14** | `Adslotvisibility`| Slot visibility (`FirstView`, `SecondView`, etc.) | `SecondView` |
| **15** | `Adslotformat` | Slot layout (`Fixed`, `Pop`, etc.) | `Fixed` |
| **16** | `Adslotfloorprice`| Minimum reserve price in CPM | `5` |
| **17** | `CreativeID` | Identifier for the ad creative | `e39e178dfd...1ee56acd` |
| **18** | `AdvertiserID` | Identifier for advertiser campaign | `3358` |
| **19** | `UserTags` | Comma-separated user interest profile tags | `10006,10024` |

---

## License & Credits

- **Code License**: Licensed under the [MIT License](LICENSE).
- **Dataset License**: Dataset belongs to IPinYou Inc. Copyright 2014 ACM 978-1-4503-2999-6/14/08. For additional permissions, contact `Permissions@acm.org`.
