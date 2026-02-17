#[derive(Debug, Clone, Copy)]
pub struct RuntimePolicy {
    pub target_fps: u32,
    pub max_stale_frames: u32,
    pub block_size: usize,
    pub cpu_threads: usize,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            target_fps: 60,
            max_stale_frames: 2,
            block_size: 65_536,
            cpu_threads: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SimulationRuntimeStats {
    pub frame_ms: f32,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    pub scheduler_block_size: usize,
    pub scheduler_quality: f32,
    pub queue_depth: usize,
    pub stale_frames: u32,
    pub approx_mode: bool,
    pub cpu_worker_utilization: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulePlan {
    pub requested_points: usize,
    pub effective_points: usize,
    pub scheduled_points: usize,
    pub gpu_points: usize,
    pub cpu_points: usize,
    pub queue_depth: usize,
    pub quality_scale: f32,
    pub approx_mode: bool,
    pub block_size: usize,
}

pub struct AdaptiveScheduler {
    policy: RuntimePolicy,
    gpu_points_per_ms_ewma: f64,
    cpu_points_per_ms_ewma: f64,
    missed_budget_streak: u32,
    approx_mode: bool,
    current_block_size: usize,
    starvation_streak: u32,
    stable_streak: u32,
    approx_recovery_streak: u32,
    quality_scale: f32,
    frame_budget_scale: f32,
}

impl AdaptiveScheduler {
    const MIN_BLOCK_SIZE: usize = 8_192;
    const MIN_QUALITY_SCALE: f32 = 0.15;
    const QUALITY_DOWN_FACTOR: f32 = 0.82;
    const QUALITY_UP_FACTOR: f32 = 1.02;
    const MIN_FRAME_BUDGET_SCALE: f32 = 0.30;
    const FRAME_BUDGET_DOWN_FACTOR: f32 = 0.90;
    const FRAME_BUDGET_UP_FACTOR: f32 = 1.01;

    pub fn new(policy: RuntimePolicy) -> Self {
        let initial_block = policy.block_size.max(1);
        Self {
            policy,
            gpu_points_per_ms_ewma: 200_000.0,
            cpu_points_per_ms_ewma: 50_000.0,
            missed_budget_streak: 0,
            approx_mode: false,
            current_block_size: initial_block,
            starvation_streak: 0,
            stable_streak: 0,
            approx_recovery_streak: 0,
            quality_scale: 1.0,
            frame_budget_scale: 1.0,
        }
    }

    pub fn policy(&self) -> RuntimePolicy {
        self.policy
    }

    pub fn plan(
        &mut self,
        requested_points: usize,
        effective_points: usize,
        queue_depth: usize,
    ) -> SchedulePlan {
        let block_size = self.current_block_size.max(1);
        let target_frame_ms = self.target_frame_ms();
        let predicted_capacity = ((self.gpu_points_per_ms_ewma + self.cpu_points_per_ms_ewma)
            * target_frame_ms)
            .max(block_size as f64) as usize;
        let request_quality_cap = request_pressure_scale(requested_points, effective_points);
        let queue_penalty = if self.approx_mode {
            (1.0 - queue_depth as f32 * 0.10).clamp(0.60, 1.0)
        } else {
            1.0
        };
        let applied_quality =
            (self.quality_scale.min(request_quality_cap) * self.frame_budget_scale * queue_penalty)
                .clamp(Self::MIN_QUALITY_SCALE, 1.0);

        let must_approximate = requested_points > effective_points;
        if must_approximate {
            self.approx_mode = true;
        }

        let mut scheduled_points = if self.approx_mode {
            let scaled_capacity = (predicted_capacity as f64 * applied_quality as f64) as usize;
            effective_points.min(scaled_capacity.max(block_size))
        } else {
            effective_points
        };

        if self.approx_mode {
            let floor = block_size.min(effective_points.max(1));
            scheduled_points = scheduled_points.max(floor);
        }

        scheduled_points = align_up(scheduled_points, block_size).min(effective_points);

        let total_rate = (self.gpu_points_per_ms_ewma + self.cpu_points_per_ms_ewma).max(1.0);
        let mut gpu_share = self.gpu_points_per_ms_ewma / total_rate;

        if queue_depth >= self.policy.max_stale_frames as usize {
            gpu_share = 0.0;
        }

        let mut gpu_points = (scheduled_points as f64 * gpu_share) as usize;
        gpu_points = align_down(gpu_points, block_size).min(scheduled_points);

        if scheduled_points > 0
            && gpu_points == 0
            && queue_depth < self.policy.max_stale_frames as usize
        {
            gpu_points = scheduled_points.min(block_size);
        }

        let cpu_points = scheduled_points.saturating_sub(gpu_points);

        SchedulePlan {
            requested_points,
            effective_points,
            scheduled_points,
            gpu_points,
            cpu_points,
            queue_depth,
            quality_scale: applied_quality,
            approx_mode: self.approx_mode,
            block_size,
        }
    }

    pub fn record_frame(&mut self, plan: SchedulePlan, gpu_ms: f64, cpu_ms: f64, frame_ms: f64) {
        if plan.gpu_points > 0 && gpu_ms > 0.0 {
            let throughput = plan.gpu_points as f64 / gpu_ms;
            self.gpu_points_per_ms_ewma = ewma(self.gpu_points_per_ms_ewma, throughput, 0.25);
        }

        if plan.cpu_points > 0 && cpu_ms > 0.0 {
            let throughput = plan.cpu_points as f64 / cpu_ms;
            self.cpu_points_per_ms_ewma = ewma(self.cpu_points_per_ms_ewma, throughput, 0.25);
        }

        if frame_ms > self.target_frame_ms() * 1.05 {
            self.missed_budget_streak = self.missed_budget_streak.saturating_add(1);
        } else {
            self.missed_budget_streak = 0;
        }

        // Escalate to approximation quickly once sustained misses are clear.
        if self.missed_budget_streak >= 2 {
            self.approx_mode = true;
            self.approx_recovery_streak = 0;
        }

        // Single-frame severe overload should also trigger approximation for
        // exact-mode requests, to protect frame pacing.
        if !self.approx_mode
            && frame_ms > self.target_frame_ms() * 1.35
            && plan.requested_points <= plan.effective_points
        {
            self.approx_mode = true;
            self.approx_recovery_streak = 0;
        }

        self.retune_frame_budget_scale(plan, frame_ms);
        self.retune_quality(plan, frame_ms);
        self.retune_block_size(plan, frame_ms);

        if self.approx_mode && plan.requested_points <= plan.effective_points {
            if frame_ms < self.target_frame_ms() * 0.70 && plan.queue_depth == 0 {
                self.approx_recovery_streak = self.approx_recovery_streak.saturating_add(1);
                if self.approx_recovery_streak >= 6 {
                    self.approx_mode = false;
                    self.approx_recovery_streak = 0;
                }
            } else {
                self.approx_recovery_streak = 0;
            }
        } else {
            self.approx_recovery_streak = 0;
        }
    }

    pub fn worker_utilization(&self, cpu_points: usize, block_size: usize) -> f32 {
        let nominal_capacity = (self.policy.cpu_threads.max(1) * block_size.max(1)) as f32;
        (cpu_points as f32 / nominal_capacity).clamp(0.0, 1.0)
    }

    fn target_frame_ms(&self) -> f64 {
        1_000.0 / self.policy.target_fps.max(1) as f64
    }

    fn min_block_size(&self) -> usize {
        Self::MIN_BLOCK_SIZE.min(self.policy.block_size.max(1))
    }

    fn retune_block_size(&mut self, plan: SchedulePlan, frame_ms: f64) {
        let target_frame_ms = self.target_frame_ms();
        let nominal_capacity = self.policy.cpu_threads.max(1) * plan.block_size.max(1);
        let worker_util = if nominal_capacity > 0 {
            plan.cpu_points as f64 / nominal_capacity as f64
        } else {
            1.0
        };

        // Under heavy load, sustained low worker utilization means blocks are
        // too coarse for full CPU participation.
        let heavy_load = plan.effective_points >= nominal_capacity.saturating_mul(2);
        let cpu_starved = heavy_load && plan.cpu_points > 0 && worker_util < 0.80;

        if cpu_starved {
            self.starvation_streak = self.starvation_streak.saturating_add(1);
        } else {
            self.starvation_streak = 0;
        }

        if self.starvation_streak >= 2 {
            let min_block = self.min_block_size();
            if self.current_block_size > min_block {
                self.current_block_size =
                    align_down((self.current_block_size / 2).max(min_block), 1_024).max(min_block);
            }
            self.starvation_streak = 0;
            self.stable_streak = 0;
            return;
        }

        // Scale block size back up gradually once frame pacing is healthy.
        let frame_stable = frame_ms < target_frame_ms * 0.90;
        if frame_stable && self.missed_budget_streak == 0 && worker_util > 0.95 {
            self.stable_streak = self.stable_streak.saturating_add(1);
        } else {
            self.stable_streak = 0;
        }

        if self.stable_streak >= 6 && self.current_block_size < self.policy.block_size {
            self.current_block_size = (self.current_block_size * 2).min(self.policy.block_size);
            self.stable_streak = 0;
        }
    }

    fn retune_quality(&mut self, plan: SchedulePlan, frame_ms: f64) {
        let target_frame_ms = self.target_frame_ms();
        let overloaded = frame_ms > target_frame_ms * 1.05;
        let queue_backed_up = plan.queue_depth >= self.policy.max_stale_frames as usize;

        if self.approx_mode && (overloaded || queue_backed_up) {
            let mut down_factor = Self::QUALITY_DOWN_FACTOR;
            if frame_ms > target_frame_ms * 1.30 {
                down_factor *= 0.80;
            }
            if queue_backed_up {
                down_factor *= 0.90;
            }
            self.quality_scale = (self.quality_scale * down_factor).max(Self::MIN_QUALITY_SCALE);
            return;
        }

        let stable = frame_ms < target_frame_ms * 0.80 && plan.queue_depth == 0;
        if stable {
            self.quality_scale = (self.quality_scale * Self::QUALITY_UP_FACTOR).min(1.0);
        }
    }

    fn retune_frame_budget_scale(&mut self, plan: SchedulePlan, frame_ms: f64) {
        let target_frame_ms = self.target_frame_ms();
        if frame_ms > target_frame_ms * 1.05 {
            let mut down = Self::FRAME_BUDGET_DOWN_FACTOR;
            if frame_ms > target_frame_ms * 1.30 {
                down *= 0.80;
            }
            if plan.queue_depth >= self.policy.max_stale_frames as usize {
                down *= 0.90;
            }
            self.frame_budget_scale =
                (self.frame_budget_scale * down).max(Self::MIN_FRAME_BUDGET_SCALE);
            return;
        }

        let stable = frame_ms < target_frame_ms * 0.78 && plan.queue_depth == 0;
        if stable {
            self.frame_budget_scale =
                (self.frame_budget_scale * Self::FRAME_BUDGET_UP_FACTOR).min(1.0);
        }
    }
}

fn ewma(current: f64, sample: f64, alpha: f64) -> f64 {
    current * (1.0 - alpha) + sample * alpha
}

fn align_up(value: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return value;
    }

    let remainder = value % block_size;
    if remainder == 0 {
        value
    } else {
        value + (block_size - remainder)
    }
}

fn align_down(value: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return value;
    }

    value - (value % block_size)
}

fn request_pressure_scale(requested_points: usize, effective_points: usize) -> f32 {
    if requested_points <= effective_points || effective_points == 0 {
        return 1.0;
    }

    let ratio = requested_points as f32 / effective_points as f32;
    ratio.sqrt().recip().clamp(0.15, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_is_block_aligned() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let plan = scheduler.plan(1_000_000, 1_000_000, 0);
        assert!(plan.scheduled_points <= plan.effective_points);
        if plan.gpu_points >= plan.block_size {
            assert_eq!(plan.gpu_points % plan.block_size, 0);
        }
        assert_eq!(plan.cpu_points + plan.gpu_points, plan.scheduled_points);
    }

    #[test]
    fn switches_to_approx_after_missed_budget_streak() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let plan = scheduler.plan(5_000_000, 5_000_000, 0);
        for _ in 0..2 {
            scheduler.record_frame(plan, 0.0, 30.0, 35.0);
        }
        let plan_after = scheduler.plan(5_000_000, 5_000_000, 0);
        assert!(plan_after.approx_mode);
    }

    #[test]
    fn severe_single_frame_overload_enables_approximation() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let plan = scheduler.plan(5_000_000, 5_000_000, 0);
        scheduler.record_frame(plan, 0.0, 30.0, 50.0);
        let plan_after = scheduler.plan(5_000_000, 5_000_000, 0);
        assert!(plan_after.approx_mode);
    }

    #[test]
    fn approximation_recovery_requires_stable_frames() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let plan = scheduler.plan(5_000_000, 5_000_000, 0);
        for _ in 0..2 {
            scheduler.record_frame(plan, 0.0, 30.0, 35.0);
        }
        assert!(scheduler.plan(5_000_000, 5_000_000, 0).approx_mode);

        let fast_frame = scheduler.plan(5_000_000, 5_000_000, 0);
        scheduler.record_frame(fast_frame, 2.0, 2.0, 8.0);
        assert!(scheduler.plan(5_000_000, 5_000_000, 0).approx_mode);

        for _ in 0..5 {
            let stable_plan = scheduler.plan(5_000_000, 5_000_000, 0);
            scheduler.record_frame(stable_plan, 2.0, 2.0, 8.0);
        }
        assert!(!scheduler.plan(5_000_000, 5_000_000, 0).approx_mode);
    }

    #[test]
    fn shrinks_block_size_when_cpu_starved_under_heavy_load() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let baseline = scheduler.plan(5_000_000, 5_000_000, 0);

        let starved_plan = SchedulePlan {
            requested_points: 5_000_000,
            effective_points: 5_000_000,
            scheduled_points: 5_000_000,
            gpu_points: 4_900_000,
            cpu_points: baseline.block_size,
            queue_depth: 0,
            quality_scale: 1.0,
            approx_mode: false,
            block_size: baseline.block_size,
        };

        scheduler.record_frame(starved_plan, 8.0, 2.0, 18.0);
        scheduler.record_frame(starved_plan, 8.0, 2.0, 18.0);

        let tuned = scheduler.plan(5_000_000, 5_000_000, 0);
        assert!(tuned.block_size < baseline.block_size);
    }

    #[test]
    fn requested_above_effective_forces_adaptive_cap() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let plan = scheduler.plan(100_000_000, 5_000_000, 0);
        assert!(plan.approx_mode);
        assert!(plan.scheduled_points <= plan.effective_points);
        assert_eq!(plan.effective_points, 5_000_000);
        assert!(plan.quality_scale <= 0.30);
    }

    #[test]
    fn quality_scale_reduces_under_sustained_pressure() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let baseline = scheduler.plan(6_000_000, 5_000_000, 0);

        for _ in 0..4 {
            let stressed = scheduler.plan(6_000_000, 5_000_000, 3);
            scheduler.record_frame(stressed, 12.0, 8.0, 30.0);
        }

        let tuned = scheduler.plan(6_000_000, 5_000_000, 3);
        assert!(tuned.quality_scale < baseline.quality_scale);
    }

    #[test]
    fn quality_scale_recovers_when_stable() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());

        for _ in 0..3 {
            let stressed = scheduler.plan(6_000_000, 5_000_000, 3);
            scheduler.record_frame(stressed, 12.0, 8.0, 30.0);
        }
        let reduced = scheduler.plan(6_000_000, 5_000_000, 0);

        for _ in 0..16 {
            let stable = scheduler.plan(6_000_000, 5_000_000, 0);
            scheduler.record_frame(stable, 2.0, 2.0, 8.0);
        }
        let recovered = scheduler.plan(6_000_000, 5_000_000, 0);

        assert!(recovered.quality_scale >= reduced.quality_scale);
    }

    #[test]
    fn request_pressure_lowers_quality_for_extreme_ratios() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let moderate = scheduler.plan(6_000_000, 5_000_000, 0);
        let extreme = scheduler.plan(100_000_000, 5_000_000, 0);

        assert!(extreme.quality_scale < moderate.quality_scale);
    }

    #[test]
    fn frame_budget_scale_reduces_when_over_budget() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());
        let baseline = scheduler.plan(6_000_000, 5_000_000, 0);

        for _ in 0..4 {
            let plan = scheduler.plan(6_000_000, 5_000_000, 0);
            scheduler.record_frame(plan, 10.0, 10.0, 35.0);
        }

        let tuned = scheduler.plan(6_000_000, 5_000_000, 0);
        assert!(tuned.quality_scale < baseline.quality_scale);
    }

    #[test]
    fn frame_budget_scale_recovers_when_stable() {
        let mut scheduler = AdaptiveScheduler::new(RuntimePolicy::default());

        for _ in 0..4 {
            let plan = scheduler.plan(6_000_000, 5_000_000, 0);
            scheduler.record_frame(plan, 10.0, 10.0, 35.0);
        }
        let reduced = scheduler.plan(6_000_000, 5_000_000, 0);

        for _ in 0..24 {
            let plan = scheduler.plan(6_000_000, 5_000_000, 0);
            scheduler.record_frame(plan, 2.0, 2.0, 8.0);
        }
        let recovered = scheduler.plan(6_000_000, 5_000_000, 0);
        assert!(recovered.quality_scale >= reduced.quality_scale);
    }
}
