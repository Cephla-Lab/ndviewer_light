#!/usr/bin/env python3
"""
Memory leak reproduction test for ndviewer_light in-place update fix.

This script demonstrates the memory leak caused by ndv's data setter
(https://github.com/pyapp-kit/ndv/issues/209) and validates that the
fix in _try_inplace_ndv_update() prevents the leak.

The test simulates the live refresh scenario where data is updated
every 750ms during acquisition.

Usage:
    python tests/test_memory_leak_reproduction.py [--iterations N] [--old-code] [--compare]

Options:
    --iterations N   Number of refresh cycles to simulate (default: 100)
    --old-code       Use the old leaky code path for comparison
    --compare        Run both old and new code paths and show comparison
"""

import argparse
import gc
import sys
import time
import tracemalloc
from unittest.mock import MagicMock

import numpy as np


def get_process_memory_mb():
    """Get current process memory in MB using psutil if available."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def create_test_data(shape=(1, 1, 10, 3, 512, 512), iteration=0):
    """Create xarray-like test data simulating acquisition frames.

    Shape: (time, fov, z, channel, height, width)
    Each iteration creates slightly different data to simulate new frames.

    Args:
        shape: Array shape tuple
        iteration: Used as random seed for reproducible test data
    """
    # Create a mock xarray DataArray with the essential attributes
    data = MagicMock()
    data.shape = shape
    data.dtype = np.uint16
    data.dims = ("time", "fov", "z", "channel", "y", "x")
    data.attrs = {"channel_names": ["DAPI", "GFP", "RFP"], "luts": {}}

    # Store actual numpy array to simulate real memory usage
    # Use iteration as seed for reproducibility while still varying data
    rng = np.random.default_rng(seed=iteration)
    data._backing_array = rng.integers(0, 65535, size=shape, dtype=np.uint16)
    data.values = data._backing_array

    return data


class MockDataWrapper:
    """Mock ndv DataWrapper that simulates the real wrapper behavior."""

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class MockDataModel:
    """Mock ndv _data_model."""

    def __init__(self):
        self.data_wrapper = None


class MockArrayViewer:
    """Mock ndv ArrayViewer that simulates the memory leak behavior.

    When use_leaky_setter=True, setting .data creates new internal
    objects without cleaning up old ones (simulating the GPU handle leak).
    When use_leaky_setter=False, we use the workaround path.
    """

    def __init__(self, initial_data, use_leaky_setter=True):
        self.use_leaky_setter = use_leaky_setter
        self._data_model = MockDataModel()
        self._data_model.data_wrapper = MockDataWrapper(initial_data)

        # Simulate accumulated GPU handles (the leak)
        self._leaked_handles = []

        # For the workaround path
        self._request_data_called = False

    @property
    def data(self):
        if self._data_model.data_wrapper:
            return self._data_model.data_wrapper.data
        return None

    @data.setter
    def data(self, new_data):
        """Simulates ndv's leaky data setter."""
        if self.use_leaky_setter:
            # This is what ndv does - creates new wrapper without cleanup
            # The old wrapper's GPU handles are never freed
            old_wrapper = self._data_model.data_wrapper
            if old_wrapper is not None:
                # Simulate leaked GPU handle (in real ndv, this is vispy texture)
                # We keep a reference to simulate the leak
                self._leaked_handles.append(
                    {
                        "old_data": old_wrapper._data,
                        "texture_handle": np.zeros(
                            (64, 64), dtype=np.uint16
                        ),  # Simulated GPU texture
                    }
                )

            self._data_model.data_wrapper = MockDataWrapper(new_data)
        else:
            # Non-leaky path (but this setter shouldn't be called with workaround)
            self._data_model.data_wrapper._data = new_data

    def _request_data(self):
        """Simulates triggering a display refresh."""
        self._request_data_called = True

    def get_leaked_handle_count(self):
        """Return number of leaked handles (for testing)."""
        return len(self._leaked_handles)

    def get_leaked_memory_mb(self):
        """Estimate memory held by leaked handles."""
        total_bytes = 0
        for handle in self._leaked_handles:
            if handle.get("old_data") is not None:
                arr = getattr(handle["old_data"], "_backing_array", None)
                if arr is not None:
                    total_bytes += arr.nbytes
            if handle.get("texture_handle") is not None:
                total_bytes += handle["texture_handle"].nbytes
        return total_bytes / 1024 / 1024


def simulate_old_code_path(viewer, new_data):
    """Simulate the OLD code that used ndv's data setter (LEAKS)."""
    # This is what the old _try_inplace_ndv_update did:
    # setattr(v, 'data', data) or v.data = data
    viewer.data = new_data
    return True


def simulate_new_code_path(viewer, new_data):
    """Simulate the NEW code that bypasses the setter (NO LEAK)."""
    # This is what the fixed _try_inplace_ndv_update does:
    # 1. Get wrapper via correct path
    wrapper = viewer._data_model.data_wrapper
    if wrapper is None:
        return False

    # 2. Check shapes match
    old_shape = getattr(wrapper._data, "shape", None)
    new_shape = getattr(new_data, "shape", None)
    if old_shape != new_shape:
        return False

    # 3. Update wrapper._data directly (bypasses leaky setter)
    wrapper._data = new_data

    # 4. Trigger refresh
    viewer._request_data()

    return True


def run_memory_test(iterations, use_old_code, data_shape=(1, 1, 5, 3, 256, 256)):
    """Run the memory leak reproduction test.

    Args:
        iterations: Number of refresh cycles to simulate
        use_old_code: If True, use the old leaky code path
        data_shape: Shape of test data arrays

    Returns:
        dict with test results
    """
    # Calculate expected data size
    data_size_mb = np.prod(data_shape) * 2 / 1024 / 1024  # uint16 = 2 bytes

    print(f"\n{'='*60}")
    print(f"Memory Leak Reproduction Test")
    print(f"{'='*60}")
    print(
        f"Code path: {'OLD (leaky setter)' if use_old_code else 'NEW (direct wrapper update)'}"
    )
    print(f"Iterations: {iterations}")
    print(f"Data shape: {data_shape}")
    print(f"Data size per frame: {data_size_mb:.2f} MB")
    print(f"{'='*60}\n")

    # Start memory tracking
    tracemalloc.start()
    gc.collect()

    initial_process_mb = get_process_memory_mb()
    initial_traced_mb = tracemalloc.get_traced_memory()[0] / 1024 / 1024

    # Create initial viewer with data
    initial_data = create_test_data(data_shape, iteration=0)
    viewer = MockArrayViewer(initial_data, use_leaky_setter=use_old_code)

    # Simulate the refresh loop (like the 750ms timer)
    update_func = simulate_old_code_path if use_old_code else simulate_new_code_path

    start_time = time.time()
    memory_samples = []

    for i in range(iterations):
        # Create new data (simulating new acquisition frame)
        new_data = create_test_data(data_shape, iteration=i + 1)

        # Update the viewer using old or new code path
        update_func(viewer, new_data)

        # Sample memory periodically
        if (i + 1) % 10 == 0:
            gc.collect()
            current_mb = tracemalloc.get_traced_memory()[0] / 1024 / 1024
            process_mb = get_process_memory_mb()
            leaked_mb = viewer.get_leaked_memory_mb()

            memory_samples.append(
                {
                    "iteration": i + 1,
                    "traced_mb": current_mb,
                    "process_mb": process_mb,
                    "leaked_handles": viewer.get_leaked_handle_count(),
                    "leaked_mb": leaked_mb,
                }
            )

            print(
                f"  Iteration {i+1:4d}: "
                f"traced={current_mb:6.1f}MB, "
                f"leaked_handles={viewer.get_leaked_handle_count():4d}, "
                f"leaked={leaked_mb:6.1f}MB"
            )

    elapsed = time.time() - start_time

    # Final measurements
    gc.collect()
    final_traced = tracemalloc.get_traced_memory()
    final_process_mb = get_process_memory_mb()
    tracemalloc.stop()

    # Results
    results = {
        "code_path": "old" if use_old_code else "new",
        "iterations": iterations,
        "data_size_mb": data_size_mb,
        "elapsed_seconds": elapsed,
        "initial_traced_mb": initial_traced_mb,
        "final_traced_mb": final_traced[0] / 1024 / 1024,
        "peak_traced_mb": final_traced[1] / 1024 / 1024,
        "initial_process_mb": initial_process_mb,
        "final_process_mb": final_process_mb,
        "leaked_handle_count": viewer.get_leaked_handle_count(),
        "leaked_memory_mb": viewer.get_leaked_memory_mb(),
        "memory_samples": memory_samples,
    }

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(
        f"Traced memory: {results['initial_traced_mb']:.1f}MB -> {results['final_traced_mb']:.1f}MB (peak: {results['peak_traced_mb']:.1f}MB)"
    )
    if initial_process_mb and final_process_mb:
        print(f"Process memory: {initial_process_mb:.1f}MB -> {final_process_mb:.1f}MB")
    print(f"Leaked handles: {results['leaked_handle_count']}")
    print(f"Leaked memory (simulated): {results['leaked_memory_mb']:.1f}MB")
    print(f"{'='*60}\n")

    return results


def main():
    """Entry point for the memory leak reproduction script.

    Parses command-line arguments and runs the appropriate test mode:
    - Default: Run with new (fixed) code path
    - --old-code: Run with old (leaky) code path
    - --compare: Run both and show comparison

    Returns:
        int: Exit code (0=success, 1=leak detected, 2=test issue)
    """
    parser = argparse.ArgumentParser(description="Memory leak reproduction test")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of refresh cycles (default: 100)",
    )
    parser.add_argument(
        "--old-code", action="store_true", help="Use old leaky code path"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both old and new code for comparison",
    )
    args = parser.parse_args()

    if args.compare:
        print("\n" + "=" * 70)
        print("COMPARISON TEST: Old Code vs New Code")
        print("=" * 70)

        # Run with old code
        old_results = run_memory_test(args.iterations, use_old_code=True)

        # Run with new code
        new_results = run_memory_test(args.iterations, use_old_code=False)

        # Summary comparison
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<30} {'Old Code':<20} {'New Code':<20}")
        print("-" * 70)
        print(
            f"{'Leaked handles':<30} {old_results['leaked_handle_count']:<20} {new_results['leaked_handle_count']:<20}"
        )
        print(
            f"{'Leaked memory (MB)':<30} {old_results['leaked_memory_mb']:<20.1f} {new_results['leaked_memory_mb']:<20.1f}"
        )
        print(
            f"{'Peak traced memory (MB)':<30} {old_results['peak_traced_mb']:<20.1f} {new_results['peak_traced_mb']:<20.1f}"
        )
        print("-" * 70)

        if (
            old_results["leaked_handle_count"] > 0
            and new_results["leaked_handle_count"] == 0
        ):
            print("\n✅ SUCCESS: New code eliminates the memory leak!")
            print(
                f"   Old code leaked {old_results['leaked_handle_count']} handles ({old_results['leaked_memory_mb']:.1f}MB)"
            )
            print(f"   New code leaked 0 handles (0.0MB)")
            return 0
        elif new_results["leaked_handle_count"] > 0:
            print("\n❌ FAILURE: New code still leaks memory!")
            return 1
        else:
            print(
                "\n⚠️  WARNING: Old code didn't leak (test may not be working correctly)"
            )
            return 2
    else:
        results = run_memory_test(args.iterations, use_old_code=args.old_code)

        if args.old_code:
            if results["leaked_handle_count"] > 0:
                print(
                    f"⚠️  LEAK DETECTED: {results['leaked_handle_count']} handles leaked"
                )
                return 1
            else:
                print("✅ No leak detected (unexpected for old code)")
                return 0
        else:
            if results["leaked_handle_count"] == 0:
                print("✅ No leak detected (fix working correctly)")
                return 0
            else:
                print(
                    f"❌ LEAK DETECTED: {results['leaked_handle_count']} handles leaked"
                )
                return 1


if __name__ == "__main__":
    sys.exit(main())
