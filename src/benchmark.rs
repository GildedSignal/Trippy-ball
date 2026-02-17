// Benchmark Utility
//
// This module provides benchmarking tools to compare CPU and GPU performance
// for wavefunction evaluation. It measures the time taken to evaluate the
// wavefunction for a large number of points using both CPU and GPU implementations.

use crate::math::{evaluate_wavefunction_batch, WavefunctionParams};
use crate::render::GpuWavefunctionEvaluator;
use crate::sim::{
    effective_point_count, generate_positions, AdaptiveScheduler, RuntimePolicy, SchedulePlan,
};
use crate::{debug, info};
use glam::Vec3;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Benchmark configuration
pub struct BenchmarkConfig {
    pub point_counts: Vec<usize>,
    pub iterations: usize,
    pub params: WavefunctionParams,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            point_counts: vec![10_000, 50_000, 100_000, 500_000, 1_000_000],
            iterations: 5,
            params: WavefunctionParams {
                n: 4,
                l: 3,
                m: 2,
                n2: 5,
                l2: 4,
                m2: 2,
                mix: 0.35,
                relative_phase: 0.0,
                z: 1.0,
                time_factor: 1.0,
            },
        }
    }
}

// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub point_count: usize,
    pub cpu_time: Duration,
    pub gpu_time: Duration,
    pub speedup: f64,
}

// Run benchmarks
pub fn run_benchmarks(config: BenchmarkConfig) -> Vec<BenchmarkResult> {
    info!("Starting benchmarks with {} iterations", config.iterations);
    info!("Running benchmarks with native Metal evaluator");

    let mut results = Vec::new();

    for &point_count in &config.point_counts {
        info!("Benchmarking with {} points", point_count);

        // Generate random points
        let positions = generate_random_points(point_count);

        // Benchmark CPU implementation
        let cpu_time = benchmark_cpu(&positions, &config.params, config.iterations);

        // Benchmark GPU implementation
        let gpu_time = benchmark_gpu(&positions, &config.params, config.iterations);

        // Calculate speedup
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        // Store results
        let result = BenchmarkResult {
            point_count,
            cpu_time,
            gpu_time,
            speedup,
        };

        info!("Results for {} points:", point_count);
        info!("  CPU time: {:?}", cpu_time);
        info!("  GPU time: {:?}", gpu_time);
        info!("  Speedup: {:.2}x", speedup);

        results.push(result);
    }

    results
}

// Generate random points for benchmarking
fn generate_random_points(count: usize) -> Vec<Vec3> {
    let mut rng = SplitMix64::new(0x9E37_79B9_7F4A_7C15 ^ count as u64);
    let mut positions = Vec::with_capacity(count);

    for _ in 0..count {
        let x = rng.next_range(-5.0, 5.0);
        let y = rng.next_range(-5.0, 5.0);
        let z = rng.next_range(-5.0, 5.0);
        positions.push(Vec3::new(x, y, z));
    }

    positions
}

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let fraction = ((self.next_u64() >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        fraction.clamp(0.0, 1.0)
    }

    fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

// Benchmark CPU implementation
fn benchmark_cpu(positions: &[Vec3], params: &WavefunctionParams, iterations: usize) -> Duration {
    let mut intensities = vec![0.0; positions.len()];
    let mut total_time = Duration::new(0, 0);

    // Warmup
    evaluate_wavefunction_batch(positions, params, 0.0, &mut intensities);

    // Benchmark
    for i in 0..iterations {
        let start = Instant::now();
        evaluate_wavefunction_batch(positions, params, i as f64 * 0.1, &mut intensities);
        let elapsed = start.elapsed();
        total_time += elapsed;
        debug!("CPU iteration {}: {:?}", i, elapsed);
    }

    total_time / iterations as u32
}

// Benchmark GPU implementation
fn benchmark_gpu(positions: &[Vec3], params: &WavefunctionParams, iterations: usize) -> Duration {
    // Create GPU evaluator
    let mut evaluator =
        GpuWavefunctionEvaluator::new_with_buffer_pool(params, positions.len(), None);

    // Update positions
    evaluator.update_positions(positions);

    // Warmup
    evaluator.compute(params, 0.0);
    let _ = evaluator.read_intensities();

    // Benchmark
    let mut total_time = Duration::new(0, 0);

    for i in 0..iterations {
        let start = Instant::now();

        // Compute wavefunction
        evaluator.compute(params, i as f64 * 0.1);

        // Read back results (include this in timing to be fair)
        let _ = evaluator.read_intensities();

        let elapsed = start.elapsed();
        total_time += elapsed;
        debug!("GPU iteration {}: {:?}", i, elapsed);
    }

    total_time / iterations as u32
}

fn submit_gpu_points(
    evaluator: &mut GpuWavefunctionEvaluator,
    params: &WavefunctionParams,
    sim_time: f64,
    requested_points: usize,
) -> usize {
    if requested_points > 0
        && evaluator.enqueue_compute_range_no_readback(params, sim_time, 0, requested_points)
    {
        requested_points
    } else {
        0
    }
}

// Run a quick benchmark and print results
pub fn quick_benchmark() {
    // Use smaller point counts for quick benchmark
    let config = BenchmarkConfig {
        point_counts: vec![10_000, 50_000, 100_000],
        iterations: 3,
        params: WavefunctionParams::default(),
    };

    let results = run_benchmarks(config);
    print_summary("Benchmark Summary", &results);
}

pub fn stress_benchmark() {
    let config = BenchmarkConfig {
        point_counts: vec![100_000, 1_000_000, 5_000_000],
        iterations: 2,
        params: WavefunctionParams::default(),
    };

    let results = run_benchmarks(config);
    print_summary("Stress Benchmark Summary", &results);
}

fn print_summary(title: &str, results: &[BenchmarkResult]) {
    println!("\n{title}:");
    println!("------------------");
    println!("| Points | CPU Time | GPU Time | Speedup |");
    println!("|--------------------|---------|---------|");

    for result in results {
        println!(
            "| {:7} | {:9?} | {:9?} | {:6.2}x |",
            result.point_count, result.cpu_time, result.gpu_time, result.speedup
        );
    }
}

#[derive(Debug, Clone)]
struct SoakConfig {
    requested_point_schedule: Vec<usize>,
    frames_per_stage: usize,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            requested_point_schedule: vec![100_000, 1_000_000, 5_000_000, 20_000_000, 100_000_000],
            frames_per_stage: 24,
        }
    }
}

#[derive(Debug, Clone)]
struct StageSoakStats {
    requested_points: usize,
    effective_points: usize,
    frame_ms_samples: Vec<f64>,
    budget_miss: usize,
    approx_frames: usize,
    max_queue_depth: usize,
    max_stale_frames: u32,
    min_quality: f32,
    max_quality: f32,
    min_block: usize,
    max_block: usize,
}

impl StageSoakStats {
    fn new(requested_points: usize, effective_points: usize) -> Self {
        Self {
            requested_points,
            effective_points,
            frame_ms_samples: Vec::new(),
            budget_miss: 0,
            approx_frames: 0,
            max_queue_depth: 0,
            max_stale_frames: 0,
            min_quality: 1.0,
            max_quality: 0.0,
            min_block: usize::MAX,
            max_block: 0,
        }
    }
}

pub fn soak_benchmark() {
    let config = SoakConfig::default();
    let policy = RuntimePolicy::default();
    let target_frame_ms = 1_000.0 / policy.target_fps.max(1) as f64;

    let mut scheduler = AdaptiveScheduler::new(policy);
    let mut position_cache: HashMap<usize, Vec<Vec3>> = HashMap::new();

    let mut params = WavefunctionParams::default();
    let mut current_requested = config.requested_point_schedule[0];
    let mut current_effective = effective_point_count(current_requested);
    let mut current_positions = position_cache
        .entry(current_effective)
        .or_insert_with(|| generate_positions(current_effective, 5.0))
        .clone();

    let mut evaluator =
        GpuWavefunctionEvaluator::new_with_buffer_pool(&params, current_effective, None);
    evaluator.update_positions(&current_positions);
    let mut current_intensities = vec![0.0f32; current_effective];

    let mut frame_ms_samples = Vec::new();
    let mut total_budget_miss = 0usize;
    let mut approx_frames = 0usize;
    let mut max_queue_depth = 0usize;
    let mut max_stale_frames = 0u32;
    let mut min_quality = 1.0f32;
    let mut max_quality = 0.0f32;
    let mut min_block = usize::MAX;
    let mut max_block = 0usize;
    let mut non_finite_values = 0usize;
    let mut stale_frames = 0u32;
    let mut intensity_sample = Vec::new();
    let mut stage_stats: Vec<StageSoakStats> = config
        .requested_point_schedule
        .iter()
        .map(|&requested| StageSoakStats::new(requested, effective_point_count(requested)))
        .collect();

    let total_frames = config.requested_point_schedule.len() * config.frames_per_stage;

    for frame in 0..total_frames {
        let stage_index = frame / config.frames_per_stage;
        let requested = config.requested_point_schedule[stage_index];
        let effective = effective_point_count(requested);
        if requested != current_requested || effective != current_effective {
            current_requested = requested;
            current_effective = effective;
            current_positions = position_cache
                .entry(current_effective)
                .or_insert_with(|| generate_positions(current_effective, 5.0))
                .clone();
            evaluator.resize(current_effective);
            evaluator.update_positions(&current_positions);
            current_intensities.resize(current_effective, 0.0);
            stale_frames = 0;
        }

        // Sweep through valid (l,m) pairs during soak to stress kernel-path transitions.
        let l = (frame / 16) % 6;
        let m_range = (2 * l + 1).max(1);
        let m = (frame % m_range) as isize - l as isize;
        params.l = l;
        params.m = m;
        params.z = 1.0 + (stage_index % 3) as f64;
        params.time_factor = 1.0 + (frame as f64 * 0.001).sin() * 0.2;
        let sim_time = frame as f64 * 0.016;

        let frame_start = Instant::now();
        let completion = evaluator.poll_completed_no_readback();
        let completed_gpu_points = completion.completed_points;
        let completed_gpu_ms = completion.gpu_ms;
        let completed_any = completion.completed_any;

        if completed_any {
            stale_frames = 0;
        } else {
            stale_frames = stale_frames.saturating_add(1).min(policy.max_stale_frames);
        }

        let queue_depth = evaluator.in_flight_count();
        max_queue_depth = max_queue_depth.max(queue_depth);
        stage_stats[stage_index].max_queue_depth =
            stage_stats[stage_index].max_queue_depth.max(queue_depth);

        let mut plan = scheduler.plan(requested, effective, queue_depth);
        if stale_frames >= policy.max_stale_frames {
            plan.gpu_points = 0;
            plan.cpu_points = plan.scheduled_points;
        }

        min_quality = min_quality.min(plan.quality_scale);
        max_quality = max_quality.max(plan.quality_scale);
        min_block = min_block.min(plan.block_size);
        max_block = max_block.max(plan.block_size);
        stage_stats[stage_index].min_quality =
            stage_stats[stage_index].min_quality.min(plan.quality_scale);
        stage_stats[stage_index].max_quality =
            stage_stats[stage_index].max_quality.max(plan.quality_scale);
        stage_stats[stage_index].min_block =
            stage_stats[stage_index].min_block.min(plan.block_size);
        stage_stats[stage_index].max_block =
            stage_stats[stage_index].max_block.max(plan.block_size);

        if plan.approx_mode || requested > effective {
            approx_frames += 1;
            stage_stats[stage_index].approx_frames += 1;
        }

        let submitted_gpu_points =
            submit_gpu_points(&mut evaluator, &params, sim_time, plan.gpu_points);

        let cpu_target_points = if submitted_gpu_points > 0 {
            plan.cpu_points
        } else {
            plan.scheduled_points
        };
        let cpu_start_idx = submitted_gpu_points.min(effective);
        let cpu_end_idx = cpu_start_idx
            .saturating_add(cpu_target_points)
            .min(effective);

        let mut cpu_ms = 0.0f64;
        if cpu_end_idx > cpu_start_idx {
            let cpu_start = Instant::now();
            evaluate_wavefunction_batch(
                &current_positions[cpu_start_idx..cpu_end_idx],
                &params,
                sim_time,
                &mut current_intensities[cpu_start_idx..cpu_end_idx],
            );
            cpu_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
            let _ = evaluator.upload_intensity_range(
                cpu_start_idx,
                &current_intensities[cpu_start_idx..cpu_end_idx],
            );
        }

        if submitted_gpu_points == 0
            && cpu_end_idx.saturating_sub(cpu_start_idx) == plan.scheduled_points
            && plan.scheduled_points > 0
        {
            stale_frames = 0;
        }
        max_stale_frames = max_stale_frames.max(stale_frames);
        stage_stats[stage_index].max_stale_frames =
            stage_stats[stage_index].max_stale_frames.max(stale_frames);

        evaluator.copy_intensity_sample(16_384, &mut intensity_sample);
        for &value in &intensity_sample {
            if !value.is_finite() {
                non_finite_values += 1;
            }
        }

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1_000.0;
        frame_ms_samples.push(frame_ms);
        stage_stats[stage_index].frame_ms_samples.push(frame_ms);
        if frame_ms > target_frame_ms * 1.05 {
            total_budget_miss += 1;
            stage_stats[stage_index].budget_miss += 1;
        }

        let feedback_plan = SchedulePlan {
            gpu_points: completed_gpu_points,
            cpu_points: cpu_end_idx.saturating_sub(cpu_start_idx),
            ..plan
        };
        scheduler.record_frame(feedback_plan, completed_gpu_ms, cpu_ms, frame_ms);
    }

    let frame_avg_ms = frame_ms_samples.iter().sum::<f64>() / frame_ms_samples.len().max(1) as f64;
    let frame_p50 = percentile_ms(&frame_ms_samples, 0.50);
    let frame_p95 = percentile_ms(&frame_ms_samples, 0.95);
    let frame_p99 = percentile_ms(&frame_ms_samples, 0.99);
    let budget_miss_pct = 100.0 * total_budget_miss as f64 / total_frames.max(1) as f64;
    let approx_pct = 100.0 * approx_frames as f64 / total_frames.max(1) as f64;

    println!("\nSoak Benchmark Summary:");
    println!("-----------------------");
    println!("Frames: {total_frames}");
    println!(
        "Requested Stages: {}",
        config
            .requested_point_schedule
            .iter()
            .map(|count| format!("{count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Frame ms avg/p50/p95/p99: {:.3} / {:.3} / {:.3} / {:.3}",
        frame_avg_ms, frame_p50, frame_p95, frame_p99
    );
    println!(
        "Budget misses (> {:.2}ms): {} ({:.1}%)",
        target_frame_ms * 1.05,
        total_budget_miss,
        budget_miss_pct
    );
    println!(
        "Approx frames: {} ({:.1}%), queue_max={}, stale_max={}",
        approx_frames, approx_pct, max_queue_depth, max_stale_frames
    );
    println!(
        "Scheduler quality min/max: {:.2} / {:.2}, block min/max: {} / {}",
        min_quality, max_quality, min_block, max_block
    );
    println!(
        "Non-finite intensity values observed: {}",
        non_finite_values
    );
    println!("Per-stage breakdown:");
    for stage in &stage_stats {
        let stage_frames = stage.frame_ms_samples.len();
        let stage_frames_nonzero = stage_frames.max(1);
        let stage_avg_ms = stage.frame_ms_samples.iter().sum::<f64>() / stage_frames_nonzero as f64;
        let stage_p95_ms = percentile_ms(&stage.frame_ms_samples, 0.95);
        let stage_miss_pct = 100.0 * stage.budget_miss as f64 / stage_frames_nonzero as f64;
        let stage_approx_pct = 100.0 * stage.approx_frames as f64 / stage_frames_nonzero as f64;
        let stage_min_block = if stage.min_block == usize::MAX {
            0
        } else {
            stage.min_block
        };
        println!(
            "  requested={} effective={} frames={} avg_ms={:.3} p95_ms={:.3} miss_pct={:.1}% approx_pct={:.1}% q={:.2}-{:.2} block={}-{} queue_max={} stale_max={}",
            stage.requested_points,
            stage.effective_points,
            stage_frames,
            stage_avg_ms,
            stage_p95_ms,
            stage_miss_pct,
            stage_approx_pct,
            stage.min_quality,
            stage.max_quality,
            stage_min_block,
            stage.max_block,
            stage.max_queue_depth,
            stage.max_stale_frames
        );
    }
}

pub fn soak_benchmark_30m() {
    let duration_secs = std::env::var("TRIPPY_BALL_SOAK_DURATION_SECS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(30 * 60)
        .clamp(10, 12 * 60 * 60);

    soak_benchmark_timed(Duration::from_secs(duration_secs));
}

pub fn soak_benchmark_timed(duration: Duration) {
    let config = SoakConfig::default();
    let policy = RuntimePolicy::default();
    let target_frame_ms = 1_000.0 / policy.target_fps.max(1) as f64;

    let stage_count = config.requested_point_schedule.len().max(1);
    let stage_duration_secs = (duration.as_secs_f64() / stage_count as f64).max(1.0);

    let mut scheduler = AdaptiveScheduler::new(policy);
    let mut position_cache: HashMap<usize, Vec<Vec3>> = HashMap::new();

    let mut params = WavefunctionParams::default();
    let mut current_requested = config.requested_point_schedule[0];
    let mut current_effective = effective_point_count(current_requested);
    let mut current_positions = position_cache
        .entry(current_effective)
        .or_insert_with(|| generate_positions(current_effective, 5.0))
        .clone();

    let mut evaluator =
        GpuWavefunctionEvaluator::new_with_buffer_pool(&params, current_effective, None);
    evaluator.update_positions(&current_positions);
    let mut current_intensities = vec![0.0f32; current_effective];

    let mut frame_ms_samples = Vec::new();
    let mut total_budget_miss = 0usize;
    let mut approx_frames = 0usize;
    let mut max_queue_depth = 0usize;
    let mut max_stale_frames = 0u32;
    let mut min_quality = 1.0f32;
    let mut max_quality = 0.0f32;
    let mut min_block = usize::MAX;
    let mut max_block = 0usize;
    let mut non_finite_values = 0usize;
    let mut stale_frames = 0u32;
    let mut intensity_sample = Vec::new();
    let mut stage_stats: Vec<StageSoakStats> = config
        .requested_point_schedule
        .iter()
        .map(|&requested| StageSoakStats::new(requested, effective_point_count(requested)))
        .collect();

    let run_start = Instant::now();
    let mut frame = 0usize;
    while run_start.elapsed() < duration {
        let elapsed_secs = run_start.elapsed().as_secs_f64();
        let stage_index = ((elapsed_secs / stage_duration_secs).floor() as usize)
            .min(stage_count.saturating_sub(1));
        let requested = config.requested_point_schedule[stage_index];
        let effective = effective_point_count(requested);
        if requested != current_requested || effective != current_effective {
            current_requested = requested;
            current_effective = effective;
            current_positions = position_cache
                .entry(current_effective)
                .or_insert_with(|| generate_positions(current_effective, 5.0))
                .clone();
            evaluator.resize(current_effective);
            evaluator.update_positions(&current_positions);
            current_intensities.resize(current_effective, 0.0);
            stale_frames = 0;
        }

        // Sweep through valid (l,m) pairs during soak to stress kernel-path transitions.
        let l = (frame / 16) % 6;
        let m_range = (2 * l + 1).max(1);
        let m = (frame % m_range) as isize - l as isize;
        params.l = l;
        params.m = m;
        params.z = 1.0 + (stage_index % 3) as f64;
        params.time_factor = 1.0 + (frame as f64 * 0.001).sin() * 0.2;
        let sim_time = frame as f64 * 0.016;

        let frame_start = Instant::now();
        let completion = evaluator.poll_completed_no_readback();
        let completed_gpu_points = completion.completed_points;
        let completed_gpu_ms = completion.gpu_ms;
        let completed_any = completion.completed_any;

        if completed_any {
            stale_frames = 0;
        } else {
            stale_frames = stale_frames.saturating_add(1).min(policy.max_stale_frames);
        }

        let queue_depth = evaluator.in_flight_count();
        max_queue_depth = max_queue_depth.max(queue_depth);
        stage_stats[stage_index].max_queue_depth =
            stage_stats[stage_index].max_queue_depth.max(queue_depth);

        let mut plan = scheduler.plan(requested, effective, queue_depth);
        if stale_frames >= policy.max_stale_frames {
            plan.gpu_points = 0;
            plan.cpu_points = plan.scheduled_points;
        }

        min_quality = min_quality.min(plan.quality_scale);
        max_quality = max_quality.max(plan.quality_scale);
        min_block = min_block.min(plan.block_size);
        max_block = max_block.max(plan.block_size);
        stage_stats[stage_index].min_quality =
            stage_stats[stage_index].min_quality.min(plan.quality_scale);
        stage_stats[stage_index].max_quality =
            stage_stats[stage_index].max_quality.max(plan.quality_scale);
        stage_stats[stage_index].min_block =
            stage_stats[stage_index].min_block.min(plan.block_size);
        stage_stats[stage_index].max_block =
            stage_stats[stage_index].max_block.max(plan.block_size);

        if plan.approx_mode || requested > effective {
            approx_frames += 1;
            stage_stats[stage_index].approx_frames += 1;
        }

        let submitted_gpu_points =
            submit_gpu_points(&mut evaluator, &params, sim_time, plan.gpu_points);

        let cpu_target_points = if submitted_gpu_points > 0 {
            plan.cpu_points
        } else {
            plan.scheduled_points
        };
        let cpu_start_idx = submitted_gpu_points.min(effective);
        let cpu_end_idx = cpu_start_idx
            .saturating_add(cpu_target_points)
            .min(effective);

        let mut cpu_ms = 0.0f64;
        if cpu_end_idx > cpu_start_idx {
            let cpu_start = Instant::now();
            evaluate_wavefunction_batch(
                &current_positions[cpu_start_idx..cpu_end_idx],
                &params,
                sim_time,
                &mut current_intensities[cpu_start_idx..cpu_end_idx],
            );
            cpu_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
            let _ = evaluator.upload_intensity_range(
                cpu_start_idx,
                &current_intensities[cpu_start_idx..cpu_end_idx],
            );
        }

        if submitted_gpu_points == 0
            && cpu_end_idx.saturating_sub(cpu_start_idx) == plan.scheduled_points
            && plan.scheduled_points > 0
        {
            stale_frames = 0;
        }
        max_stale_frames = max_stale_frames.max(stale_frames);
        stage_stats[stage_index].max_stale_frames =
            stage_stats[stage_index].max_stale_frames.max(stale_frames);

        evaluator.copy_intensity_sample(16_384, &mut intensity_sample);
        for &value in &intensity_sample {
            if !value.is_finite() {
                non_finite_values += 1;
            }
        }

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1_000.0;
        frame_ms_samples.push(frame_ms);
        stage_stats[stage_index].frame_ms_samples.push(frame_ms);
        if frame_ms > target_frame_ms * 1.05 {
            total_budget_miss += 1;
            stage_stats[stage_index].budget_miss += 1;
        }

        let feedback_plan = SchedulePlan {
            gpu_points: completed_gpu_points,
            cpu_points: cpu_end_idx.saturating_sub(cpu_start_idx),
            ..plan
        };
        scheduler.record_frame(feedback_plan, completed_gpu_ms, cpu_ms, frame_ms);
        frame = frame.saturating_add(1);
    }

    let frame_count = frame.max(1);
    let frame_avg_ms = frame_ms_samples.iter().sum::<f64>() / frame_count as f64;
    let frame_p50 = percentile_ms(&frame_ms_samples, 0.50);
    let frame_p95 = percentile_ms(&frame_ms_samples, 0.95);
    let frame_p99 = percentile_ms(&frame_ms_samples, 0.99);
    let budget_miss_pct = 100.0 * total_budget_miss as f64 / frame_count as f64;
    let approx_pct = 100.0 * approx_frames as f64 / frame_count as f64;

    println!("\nTimed Soak Benchmark Summary:");
    println!("----------------------------");
    println!(
        "Duration target/actual: {:.1}s / {:.1}s",
        duration.as_secs_f64(),
        run_start.elapsed().as_secs_f64()
    );
    println!("Frames: {frame_count}");
    println!(
        "Requested Stages: {}",
        config
            .requested_point_schedule
            .iter()
            .map(|count| format!("{count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Stage duration target: {:.1}s", stage_duration_secs);
    println!(
        "Frame ms avg/p50/p95/p99: {:.3} / {:.3} / {:.3} / {:.3}",
        frame_avg_ms, frame_p50, frame_p95, frame_p99
    );
    println!(
        "Budget misses (> {:.2}ms): {} ({:.1}%)",
        target_frame_ms * 1.05,
        total_budget_miss,
        budget_miss_pct
    );
    println!(
        "Approx frames: {} ({:.1}%), queue_max={}, stale_max={}",
        approx_frames, approx_pct, max_queue_depth, max_stale_frames
    );
    println!(
        "Scheduler quality min/max: {:.2} / {:.2}, block min/max: {} / {}",
        min_quality, max_quality, min_block, max_block
    );
    println!(
        "Non-finite intensity values observed: {}",
        non_finite_values
    );
    println!("Per-stage breakdown:");
    for stage in &stage_stats {
        let stage_frames = stage.frame_ms_samples.len();
        let stage_frames_nonzero = stage_frames.max(1);
        let stage_avg_ms = stage.frame_ms_samples.iter().sum::<f64>() / stage_frames_nonzero as f64;
        let stage_p95_ms = percentile_ms(&stage.frame_ms_samples, 0.95);
        let stage_miss_pct = 100.0 * stage.budget_miss as f64 / stage_frames_nonzero as f64;
        let stage_approx_pct = 100.0 * stage.approx_frames as f64 / stage_frames_nonzero as f64;
        let stage_min_block = if stage.min_block == usize::MAX {
            0
        } else {
            stage.min_block
        };
        println!(
            "  requested={} effective={} frames={} avg_ms={:.3} p95_ms={:.3} miss_pct={:.1}% approx_pct={:.1}% q={:.2}-{:.2} block={}-{} queue_max={} stale_max={}",
            stage.requested_points,
            stage.effective_points,
            stage_frames,
            stage_avg_ms,
            stage_p95_ms,
            stage_miss_pct,
            stage_approx_pct,
            stage.min_quality,
            stage.max_quality,
            stage_min_block,
            stage.max_block,
            stage.max_queue_depth,
            stage.max_stale_frames
        );
    }
}

#[derive(Debug, Clone)]
struct CapSweepResult {
    cap_points: usize,
    frame_avg_ms: f64,
    frame_p95_ms: f64,
    frame_p99_ms: f64,
    budget_miss_pct: f64,
    approx_pct: f64,
    max_queue_depth: usize,
    max_stale_frames: u32,
}

pub fn cap_sweep_benchmark() {
    let caps = resolve_cap_sweep_caps();
    println!("\nCap Sweep Benchmark Summary:");
    println!("---------------------------");
    println!(
        "Caps: {}",
        caps.iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "| Effective Cap | Avg ms | P95 ms | P99 ms | Miss % | Approx % | Queue Max | Stale Max |"
    );
    println!(
        "|---------------|--------|--------|--------|--------|----------|-----------|-----------|"
    );

    for cap in caps {
        let result = run_soak_with_cap(cap);
        println!(
            "| {:13} | {:6.3} | {:6.3} | {:6.3} | {:6.1} | {:8.1} | {:9} | {:9} |",
            result.cap_points,
            result.frame_avg_ms,
            result.frame_p95_ms,
            result.frame_p99_ms,
            result.budget_miss_pct,
            result.approx_pct,
            result.max_queue_depth,
            result.max_stale_frames
        );
    }
}

fn resolve_cap_sweep_caps() -> Vec<usize> {
    const DEFAULT_CAPS: [usize; 3] = [2_000_000, 5_000_000, 10_000_000];
    const MIN_CAP: usize = 1_000;
    const MAX_CAP: usize = 20_000_000;

    parse_cap_list(
        "TRIPPY_BALL_CAP_SWEEP_CAPS",
        &DEFAULT_CAPS,
        MIN_CAP,
        MAX_CAP,
    )
}

#[derive(Debug, Clone)]
struct CapPairSweepResult {
    simulation_cap_points: usize,
    shadow_cap_points: usize,
    coupled_effective_cap_points: usize,
    frame_avg_ms: f64,
    frame_p95_ms: f64,
    frame_p99_ms: f64,
    budget_miss_pct: f64,
    approx_pct: f64,
    max_queue_depth: usize,
    max_stale_frames: u32,
}

pub fn cap_pair_sweep_benchmark() {
    let simulation_caps = resolve_cap_pair_sim_caps();
    let shadow_caps = resolve_cap_pair_shadow_caps();
    println!("\nCap Pair Sweep Benchmark Summary:");
    println!("--------------------------------");
    println!(
        "Simulation caps: {}",
        simulation_caps
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Shadow caps: {}",
        shadow_caps
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "| Sim Cap | Shadow Cap | Coupled Cap | Avg ms | P95 ms | P99 ms | Miss % | Approx % | Queue Max | Stale Max |"
    );
    println!(
        "|---------|------------|-------------|--------|--------|--------|--------|----------|-----------|-----------|"
    );

    for simulation_cap in &simulation_caps {
        for shadow_cap in &shadow_caps {
            let result = run_soak_with_cap_pair(*simulation_cap, *shadow_cap);
            println!(
                "| {:7} | {:10} | {:11} | {:6.3} | {:6.3} | {:6.3} | {:6.1} | {:8.1} | {:9} | {:9} |",
                result.simulation_cap_points,
                result.shadow_cap_points,
                result.coupled_effective_cap_points,
                result.frame_avg_ms,
                result.frame_p95_ms,
                result.frame_p99_ms,
                result.budget_miss_pct,
                result.approx_pct,
                result.max_queue_depth,
                result.max_stale_frames
            );
        }
    }
}

fn resolve_cap_pair_sim_caps() -> Vec<usize> {
    const DEFAULT_CAPS: [usize; 3] = [2_000_000, 5_000_000, 10_000_000];
    const MIN_CAP: usize = 1_000;
    const MAX_CAP: usize = 20_000_000;
    parse_cap_list(
        "TRIPPY_BALL_CAP_PAIR_SWEEP_SIM_CAPS",
        &DEFAULT_CAPS,
        MIN_CAP,
        MAX_CAP,
    )
}

fn resolve_cap_pair_shadow_caps() -> Vec<usize> {
    const DEFAULT_CAPS: [usize; 3] = [200_000, 1_000_000, 5_000_000];
    const MIN_CAP: usize = 1_000;
    const MAX_CAP: usize = 5_000_000;
    parse_cap_list(
        "TRIPPY_BALL_CAP_PAIR_SWEEP_SHADOW_CAPS",
        &DEFAULT_CAPS,
        MIN_CAP,
        MAX_CAP,
    )
}

fn parse_cap_list(env_key: &str, defaults: &[usize], min_cap: usize, max_cap: usize) -> Vec<usize> {
    let parsed = std::env::var(env_key)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .map(|cap| cap.clamp(min_cap, max_cap))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut caps = if parsed.is_empty() {
        defaults.to_vec()
    } else {
        parsed
    };
    caps.sort_unstable();
    caps.dedup();
    caps
}

fn run_soak_with_cap_pair(
    simulation_cap_points: usize,
    shadow_cap_points: usize,
) -> CapPairSweepResult {
    let coupled_effective_cap_points = simulation_cap_points.min(shadow_cap_points).max(1_000);
    let result = run_soak_with_cap(coupled_effective_cap_points);
    CapPairSweepResult {
        simulation_cap_points,
        shadow_cap_points,
        coupled_effective_cap_points,
        frame_avg_ms: result.frame_avg_ms,
        frame_p95_ms: result.frame_p95_ms,
        frame_p99_ms: result.frame_p99_ms,
        budget_miss_pct: result.budget_miss_pct,
        approx_pct: result.approx_pct,
        max_queue_depth: result.max_queue_depth,
        max_stale_frames: result.max_stale_frames,
    }
}

fn run_soak_with_cap(cap_points: usize) -> CapSweepResult {
    let config = SoakConfig {
        requested_point_schedule: vec![100_000, 1_000_000, 5_000_000, 20_000_000, 100_000_000],
        frames_per_stage: 16,
    };
    let policy = RuntimePolicy::default();
    let target_frame_ms = 1_000.0 / policy.target_fps.max(1) as f64;

    let mut scheduler = AdaptiveScheduler::new(policy);
    let mut position_cache: HashMap<usize, Vec<Vec3>> = HashMap::new();
    let mut params = WavefunctionParams::default();

    let mut current_requested = config.requested_point_schedule[0];
    let mut current_effective = effective_for_cap(current_requested, cap_points);
    let mut current_positions = position_cache
        .entry(current_effective)
        .or_insert_with(|| generate_positions(current_effective, 5.0))
        .clone();

    let mut evaluator =
        GpuWavefunctionEvaluator::new_with_buffer_pool(&params, current_effective, None);
    evaluator.update_positions(&current_positions);
    let mut current_intensities = vec![0.0f32; current_effective];
    let mut intensity_sample = Vec::new();

    let mut frame_ms_samples = Vec::new();
    let mut budget_miss = 0usize;
    let mut approx_frames = 0usize;
    let mut max_queue_depth = 0usize;
    let mut max_stale_frames = 0u32;
    let mut stale_frames = 0u32;

    let total_frames = config.requested_point_schedule.len() * config.frames_per_stage;

    for frame in 0..total_frames {
        let stage_index = frame / config.frames_per_stage;
        let requested = config.requested_point_schedule[stage_index];
        let effective = effective_for_cap(requested, cap_points);

        if requested != current_requested || effective != current_effective {
            current_requested = requested;
            current_effective = effective;
            current_positions = position_cache
                .entry(current_effective)
                .or_insert_with(|| generate_positions(current_effective, 5.0))
                .clone();
            evaluator.resize(current_effective);
            evaluator.update_positions(&current_positions);
            current_intensities.resize(current_effective, 0.0);
            stale_frames = 0;
        }

        let l = (frame / 16) % 6;
        let m_range = (2 * l + 1).max(1);
        let m = (frame % m_range) as isize - l as isize;
        params.l = l;
        params.m = m;
        params.z = 1.0 + (stage_index % 3) as f64;
        params.time_factor = 1.0 + (frame as f64 * 0.001).sin() * 0.2;
        let sim_time = frame as f64 * 0.016;

        let frame_start = Instant::now();
        let completion = evaluator.poll_completed_no_readback();
        let completed_any = completion.completed_any;
        let completed_gpu_points = completion.completed_points;
        let completed_gpu_ms = completion.gpu_ms;

        if completed_any {
            stale_frames = 0;
        } else {
            stale_frames = stale_frames.saturating_add(1).min(policy.max_stale_frames);
        }

        let queue_depth = evaluator.in_flight_count();
        max_queue_depth = max_queue_depth.max(queue_depth);

        let mut plan = scheduler.plan(requested, effective, queue_depth);
        if stale_frames >= policy.max_stale_frames {
            plan.gpu_points = 0;
            plan.cpu_points = plan.scheduled_points;
        }
        if plan.approx_mode || requested > effective {
            approx_frames += 1;
        }

        let submitted_gpu_points =
            submit_gpu_points(&mut evaluator, &params, sim_time, plan.gpu_points);

        let cpu_target_points = if submitted_gpu_points > 0 {
            plan.cpu_points
        } else {
            plan.scheduled_points
        };
        let cpu_start_idx = submitted_gpu_points.min(effective);
        let cpu_end_idx = cpu_start_idx
            .saturating_add(cpu_target_points)
            .min(effective);

        let mut cpu_ms = 0.0f64;
        if cpu_end_idx > cpu_start_idx {
            let cpu_start = Instant::now();
            evaluate_wavefunction_batch(
                &current_positions[cpu_start_idx..cpu_end_idx],
                &params,
                sim_time,
                &mut current_intensities[cpu_start_idx..cpu_end_idx],
            );
            cpu_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
            let _ = evaluator.upload_intensity_range(
                cpu_start_idx,
                &current_intensities[cpu_start_idx..cpu_end_idx],
            );
        }

        if submitted_gpu_points == 0
            && cpu_end_idx.saturating_sub(cpu_start_idx) == plan.scheduled_points
            && plan.scheduled_points > 0
        {
            stale_frames = 0;
        }
        max_stale_frames = max_stale_frames.max(stale_frames);

        evaluator.copy_intensity_sample(4_096, &mut intensity_sample);
        let _sample_non_finite = intensity_sample.iter().any(|value| !value.is_finite());

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1_000.0;
        frame_ms_samples.push(frame_ms);
        if frame_ms > target_frame_ms * 1.05 {
            budget_miss += 1;
        }

        let feedback_plan = SchedulePlan {
            gpu_points: completed_gpu_points,
            cpu_points: cpu_end_idx.saturating_sub(cpu_start_idx),
            ..plan
        };
        scheduler.record_frame(feedback_plan, completed_gpu_ms, cpu_ms, frame_ms);
    }

    let total_frames_nonzero = total_frames.max(1) as f64;
    CapSweepResult {
        cap_points,
        frame_avg_ms: frame_ms_samples.iter().sum::<f64>() / frame_ms_samples.len().max(1) as f64,
        frame_p95_ms: percentile_ms(&frame_ms_samples, 0.95),
        frame_p99_ms: percentile_ms(&frame_ms_samples, 0.99),
        budget_miss_pct: 100.0 * budget_miss as f64 / total_frames_nonzero,
        approx_pct: 100.0 * approx_frames as f64 / total_frames_nonzero,
        max_queue_depth,
        max_stale_frames,
    }
}

fn effective_for_cap(requested: usize, cap_points: usize) -> usize {
    requested.clamp(1_000, cap_points.max(1_000))
}

fn percentile_ms(samples: &[f64], percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    sorted[idx]
}
