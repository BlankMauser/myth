//! Thread-scoped allocation baseline for exact RDG DrawPlan demand workloads.
//!
//! This is deliberately separate from Criterion so its allocator wrapper
//! cannot perturb wall-time measurements. Run with:
//! `cargo bench --bench rdg_allocation_probe --features rdg_bench_support`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use myth_render::graph::bench_support::{
    DrawDemandBenchHarness, DrawDemandBenchSpec, DrawDemandDependencyPattern, DrawDemandRoot,
};

#[global_allocator]
static ALLOCATOR: ThreadScopedAllocator = ThreadScopedAllocator;

struct ThreadScopedAllocator;

unsafe impl GlobalAlloc for ThreadScopedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the exact layout to the process System allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(|stats| {
                stats.alloc_calls += 1;
                stats.allocated_bytes += layout.size() as u64;
            });
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the exact layout to the process System allocator.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(|stats| {
                stats.alloc_zeroed_calls += 1;
                stats.allocated_bytes += layout.size() as u64;
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(|stats| {
            stats.dealloc_calls += 1;
            stats.deallocated_bytes += layout.size() as u64;
        });
        // SAFETY: `ptr` was allocated by System with the matching layout.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Delegates the original pointer/layout and requested size to System.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record(|stats| {
                stats.realloc_calls += 1;
                stats.reallocated_old_bytes += layout.size() as u64;
                stats.reallocated_new_bytes += new_size as u64;
            });
        }
        new_ptr
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AllocationStats {
    alloc_calls: u64,
    alloc_zeroed_calls: u64,
    realloc_calls: u64,
    dealloc_calls: u64,
    allocated_bytes: u64,
    deallocated_bytes: u64,
    reallocated_old_bytes: u64,
    reallocated_new_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CounterState {
    active: bool,
    stats: AllocationStats,
}

thread_local! {
    static COUNTER: Cell<CounterState> = const { Cell::new(CounterState {
        active: false,
        stats: AllocationStats {
            alloc_calls: 0,
            alloc_zeroed_calls: 0,
            realloc_calls: 0,
            dealloc_calls: 0,
            allocated_bytes: 0,
            deallocated_bytes: 0,
            reallocated_old_bytes: 0,
            reallocated_new_bytes: 0,
        },
    }) };
}

fn record(update: impl FnOnce(&mut AllocationStats)) {
    let _ = COUNTER.try_with(|counter| {
        let mut state = counter.get();
        if state.active {
            update(&mut state.stats);
            counter.set(state);
        }
    });
}

struct CountingScope {
    active: bool,
}

impl CountingScope {
    fn begin() -> Self {
        COUNTER.with(|counter| {
            let state = counter.get();
            assert!(!state.active, "allocation measurement scopes cannot nest");
            counter.set(CounterState {
                active: true,
                stats: AllocationStats::default(),
            });
        });
        Self { active: true }
    }

    fn finish(mut self) -> AllocationStats {
        let stats = disable_counter();
        self.active = false;
        stats
    }
}

impl Drop for CountingScope {
    fn drop(&mut self) {
        if self.active {
            let _ = disable_counter();
        }
    }
}

fn disable_counter() -> AllocationStats {
    COUNTER.with(|counter| {
        let mut state = counter.get();
        assert!(state.active, "allocation counter was not active");
        state.active = false;
        counter.set(state);
        state.stats
    })
}

fn measure<T: Copy>(operation: impl FnOnce() -> T) -> (T, AllocationStats) {
    // Prime native TLS before enabling the allocator callback.
    COUNTER.with(|counter| black_box(counter.get()));
    let scope = CountingScope::begin();
    let value = operation();
    let stats = scope.finish();
    (value, stats)
}

fn main() {
    verify_counter_is_live();
    println!(
        "RDG full-frame allocation baseline (M1 removes demand-generated churn; flat topology storage is the final zero-allocation gate)"
    );
    println!(
        "dealloc columns include preceding same-shape topology retirement and current-frame temporaries; bytes are diagnostic, not a leak/net-balance metric"
    );
    println!(
        "pattern plans repeats alloc zeroed realloc dealloc alloc_bytes realloc_old realloc_new dealloc_bytes"
    );

    for pattern in [
        DrawDemandDependencyPattern::Shared,
        DrawDemandDependencyPattern::Disjoint,
    ] {
        for plan_count in [32, 64, 128] {
            run_scenario(pattern, plan_count, 1);
        }
    }
    for repetitions in [2, 4] {
        run_scenario(DrawDemandDependencyPattern::Shared, 64, repetitions);
    }
}

fn verify_counter_is_live() {
    let ((), stats) = measure(|| {
        let mut values = Vec::with_capacity(1);
        values.push(1_u64);
        values.reserve_exact(1024);
        black_box(values);
    });
    assert!(
        stats.alloc_calls + stats.alloc_zeroed_calls + stats.realloc_calls > 0,
        "allocation counter self-test observed no allocation"
    );
    assert!(
        stats.dealloc_calls > 0,
        "allocation counter self-test observed no deallocation"
    );
}

fn run_scenario(
    pattern: DrawDemandDependencyPattern,
    plan_count: usize,
    demand_repetitions: usize,
) {
    let spec = DrawDemandBenchSpec {
        plan_count,
        textures_per_plan: 8,
        dependency_pattern: pattern,
        demand_repetitions,
        compile_topology: true,
        root: DrawDemandRoot::Both,
    };
    let mut harness = DrawDemandBenchHarness::new();
    for _ in 0..3 {
        black_box(harness.run_iteration(spec));
    }

    let mut baseline = None;
    let mut token = None;
    for _ in 0..3 {
        let (current_token, stats) = measure(|| black_box(harness.run_iteration(spec)));
        if let Some(expected) = baseline {
            assert_eq!(stats, expected, "steady allocation counts did not plateau");
        } else {
            baseline = Some(stats);
        }
        token = Some(current_token);
    }
    let stats = baseline.expect("allocation baseline was not sampled");
    let token = token.expect("DrawPlan workload was not sampled");
    assert_eq!(token.registered_pass_count, 4);
    assert_eq!(token.compiled_pass_count, 4);
    assert_eq!(token.d2_demand_count + token.cube_demand_count, plan_count);

    println!(
        "{} {} {} {} {} {} {} {} {} {} {}",
        pattern.label(),
        plan_count,
        demand_repetitions,
        stats.alloc_calls,
        stats.alloc_zeroed_calls,
        stats.realloc_calls,
        stats.dealloc_calls,
        stats.allocated_bytes,
        stats.reallocated_old_bytes,
        stats.reallocated_new_bytes,
        stats.deallocated_bytes,
    );
}
