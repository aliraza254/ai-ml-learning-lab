"""
01_idiomatic_python.py
-----------------------------------------------------------------------------
Hands-On Modern Python Idioms for AI & Machine Learning Engineers.

Key Topics:
1. Dataclasses & Type Hints for Clean ML Data Containers
2. Generators & Memory Efficiency (Batching large datasets)
3. List, Dict & Set Comprehensions
4. Custom Context Manager for Benchmarking Model & Data Processing
-----------------------------------------------------------------------------
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Generator
import time
from contextlib import contextmanager

# Ensure UTF-8 output encoding for cross-platform terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Timer Context Manager (Essential for profiling ML training & inference)
# ---------------------------------------------------------------------------
@contextmanager
def benchmark_timer(block_name: str):
    """Context manager to measure execution time of any code block."""
    start_time = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start_time) * 1000  # milliseconds
    print(f"[*] [{block_name}] completed in {elapsed:.3f} ms")


# ---------------------------------------------------------------------------
# 2. Dataclasses: Professional Data Structures for ML Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CustomerRecord:
    """Represents an immutable customer record for churn prediction."""
    customer_id: str
    age: int
    tenure_months: int
    monthly_spend: float
    support_tickets: int
    churned: bool = False
    tags: List[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        """Domain logic: high support tickets + high spend = risk signal."""
        return self.support_tickets >= 3 and self.monthly_spend > 50.0


# ---------------------------------------------------------------------------
# 3. Generators: Stream Batches Without Maxing Out RAM
# ---------------------------------------------------------------------------
def batch_generator(
    data: List[CustomerRecord], batch_size: int
) -> Generator[List[CustomerRecord], None, None]:
    """
    Yields successive batches of data.
    Generators do not hold all batches in memory simultaneously.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


# ---------------------------------------------------------------------------
# 4. Main Demonstration & Hands-On Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(">> MODULE 01: MODERN IDIOMATIC PYTHON FOR AI / ML")
    print("=" * 65)

    # A. Generate dummy records using list comprehensions
    with benchmark_timer("Data Generation (10,000 records)"):
        dataset: List[CustomerRecord] = [
            CustomerRecord(
                customer_id=f"CUST-{idx:05d}",
                age=20 + (idx % 50),
                tenure_months=(idx % 60),
                monthly_spend=25.0 + (idx % 150),
                support_tickets=(idx % 7),
                churned=(idx % 5 == 0),
            )
            for idx in range(10_000)
        ]

    print(f"[OK] Generated {len(dataset)} records.")
    print(f"[Sample] {dataset[0]}\n")

    # B. Comprehensions & Filtering (Pythonic vs Traditional Loops)
    with benchmark_timer("Filtering High-Risk Customers"):
        high_risk_customers = [c for c in dataset if c.is_high_risk]
        churn_rate_by_risk = {
            "high_risk_count": len(high_risk_customers),
            "high_risk_churned": sum(1 for c in high_risk_customers if c.churned),
        }

    print(f"[Analysis] High-Risk Analysis: {churn_rate_by_risk}")
    pct = (churn_rate_by_risk['high_risk_churned'] / churn_rate_by_risk['high_risk_count']) * 100
    print(f"[Stats] Churn percentage among high risk: {pct:.1f}%\n")

    # C. Batch Processing with Generators
    batch_size = 64
    print(f"[Batching] Streaming data in mini-batches of size {batch_size}...")
    batches_processed = 0
    total_processed_samples = 0

    with benchmark_timer("Batch Streaming Pipeline"):
        for batch in batch_generator(dataset, batch_size=batch_size):
            # Simulate a forward pass or mini-batch inference
            batches_processed += 1
            total_processed_samples += len(batch)

    print(f"[OK] Processed {batches_processed} batches ({total_processed_samples} items total).")
    print("=" * 65)
    print(">> Lesson 1 complete! Clean dataclasses, generators, and timers.")


if __name__ == "__main__":
    main()
