// Point cloud runtime state.
//
// This module now stores simulation-facing point data (positions, normalized
// intensities, visibility stats).

use crate::render::Camera;
use crate::sim::{effective_point_count, generate_positions};
use crate::{debug, warn};
use glam::Vec3;
use std::sync::atomic::{AtomicUsize, Ordering};

// Static counters for optimization statistics
static CACHED_UPDATES_COUNT: AtomicUsize = AtomicUsize::new(0);
static RECALCULATIONS_COUNT: AtomicUsize = AtomicUsize::new(0);

// Point cloud implementation
pub struct PointCloud {
    point_count: usize,
    requested_point_count: usize,
    effective_cap_override: Option<usize>,
    max_radius: f32,
    // Position buffers for simulation/debug processing
    positions: Vec<Vec3>,
    // Visibility flags for frustum culling
    visibility: Vec<bool>,
    // Visible point indices for culling statistics
    visible_indices: Vec<u32>,
    // Culling statistics
    culled_points_count: usize,
    // Whether visibility flags are current for block-priority scheduling
    visibility_valid: bool,
    // Whether frustum culling is enabled
    use_frustum_culling: bool,
    // Kept for existing debug UI contract
    points_initialized: bool,
    // Cached intensity stats to avoid O(n) scans every frame when querying debug info
    max_intensity: f32,
    min_intensity: f32,
    avg_intensity: f32,
    zero_intensity_count: usize,
}

impl PointCloud {
    const CULL_MARGIN: f32 = 0.03;
    const CULL_SAMPLE_POINTS: usize = 32_768;
    const CULL_MIN_RATIO_FOR_FULL_PASS: f32 = 0.80;
    const CULL_DISABLE_INSIDE_RADIUS_FACTOR: f32 = 1.10;
    const INTENSITY_STATS_SAMPLE_POINTS: usize = 131_072;
    const COARSE_BLOCK_SAMPLE_POINTS: usize = 512;

    // Create a new point cloud
    pub fn new(point_count: usize) -> Self {
        let requested_point_count = point_count;
        let point_count = effective_point_count(point_count);
        let max_radius = 5.0f32;
        let positions = generate_positions(point_count, max_radius);

        Self {
            point_count,
            requested_point_count,
            effective_cap_override: None,
            max_radius,
            positions,
            visibility: vec![true; point_count],
            visible_indices: (0..point_count as u32).collect(),
            culled_points_count: 0,
            visibility_valid: false,
            use_frustum_culling: true,
            points_initialized: true,
            max_intensity: 0.0,
            min_intensity: 0.0,
            avg_intensity: 0.0,
            zero_intensity_count: point_count,
        }
    }

    // Add debug visualization methods
    pub fn get_debug_info(&self) -> PointCloudDebugInfo {
        PointCloudDebugInfo {
            point_count: self.requested_point_count,
            effective_point_count: self.point_count,
            // Keep existing UI expectations for normalized debug values.
            max_intensity: self.max_intensity.max(0.1),
            min_intensity: self.min_intensity,
            avg_intensity: self.avg_intensity,
            zero_intensity_count: if self.zero_intensity_count < self.point_count / 10 {
                0
            } else {
                self.zero_intensity_count
            },
            max_radius: self.max_radius,
            points_initialized: self.points_initialized,
            cached_updates_count: CACHED_UPDATES_COUNT.load(Ordering::Relaxed),
            recalculations_count: RECALCULATIONS_COUNT.load(Ordering::Relaxed),
            culled_points_count: self.culled_points_count,
            culling_percentage: if self.point_count > 0 {
                100.0 * self.culled_points_count as f32 / self.point_count as f32
            } else {
                0.0
            },
            frame_ms: 0.0,
            gpu_ms: 0.0,
            cpu_ms: 0.0,
            scheduler_block_size: 0,
            scheduler_quality: 1.0,
            queue_depth: 0,
            stale_frames: 0,
            approx_mode: self.requested_point_count != self.point_count,
            cpu_worker_utilization: 0.0,
            native_input_points: 0,
            native_visible_points: 0,
            native_culled_points: 0,
            native_culling_percentage: 0.0,
            native_truncated_points: 0,
        }
    }

    // Update visibility based on frustum culling
    pub fn update_visibility(&mut self, camera: &Camera) {
        if !self.use_frustum_culling {
            self.visible_indices.clear();
            self.culled_points_count = 0;
            self.visibility_valid = false;
            return;
        }

        let frustum = camera.get_frustum();

        // Disable culling when camera is inside/near the sampled sphere.
        if camera.position().length() <= self.max_radius * Self::CULL_DISABLE_INSIDE_RADIUS_FACTOR {
            self.visible_indices.clear();
            self.culled_points_count = 0;
            self.visibility_valid = false;
            return;
        }

        // Whole-sphere inside frustum => no culling benefit.
        if frustum.sphere_fully_inside(Vec3::ZERO, self.max_radius) {
            self.visible_indices.clear();
            self.culled_points_count = 0;
            self.visibility_valid = false;
            return;
        }

        // Cheap estimation pass to avoid O(N) culling when expected gains are small.
        let sample_points = self.point_count.min(Self::CULL_SAMPLE_POINTS);
        let step = (self.point_count / sample_points.max(1)).max(1);
        let mut sampled = 0usize;
        let mut sampled_culled = 0usize;

        let mut i = 0usize;
        while i < self.point_count && sampled < sample_points {
            if !frustum.contains_point_with_margin(self.positions[i], Self::CULL_MARGIN) {
                sampled_culled += 1;
            }
            sampled += 1;
            i = i.saturating_add(step);
        }

        let sampled_cull_ratio = if sampled > 0 {
            sampled_culled as f32 / sampled as f32
        } else {
            0.0
        };

        if sampled_cull_ratio < Self::CULL_MIN_RATIO_FOR_FULL_PASS {
            self.visible_indices.clear();
            self.culled_points_count = 0;
            self.visibility_valid = false;
            return;
        }

        self.visible_indices.clear();
        self.visible_indices.reserve(self.point_count);

        let mut culled_count = 0usize;
        for i in 0..self.point_count {
            let visible = frustum.contains_point_with_margin(self.positions[i], Self::CULL_MARGIN);
            if visible {
                self.visible_indices.push(i as u32);
                self.visibility[i] = true;
            } else {
                culled_count += 1;
                self.visibility[i] = false;
            }
        }

        self.culled_points_count = culled_count;
        self.visibility_valid = true;
        if culled_count == 0 {
            self.visible_indices.clear();
        }
    }

    // Change the number of points
    pub fn set_point_count(&mut self, count: usize) {
        let effective_count = self.resolve_effective_count(count);

        if count == self.requested_point_count && effective_count == self.point_count {
            return;
        }

        debug!(
            "Changing point count from requested={} effective={} to requested={} effective={}",
            self.requested_point_count, self.point_count, count, effective_count
        );

        self.rebuild_point_state(count, effective_count);
    }

    pub fn set_effective_cap_override(&mut self, cap: Option<usize>) {
        let normalized_cap = cap.map(|value| value.max(1_000));
        if normalized_cap == self.effective_cap_override {
            return;
        }
        self.effective_cap_override = normalized_cap;
        let requested_count = self.requested_point_count;
        let effective_count = self.resolve_effective_count(requested_count);
        if effective_count == self.point_count {
            return;
        }
        debug!(
            "Applying runtime effective cap override: requested={} effective={}",
            requested_count, effective_count
        );
        self.rebuild_point_state(requested_count, effective_count);
    }

    fn resolve_effective_count(&self, requested_count: usize) -> usize {
        let mut effective_count = effective_point_count(requested_count);
        if let Some(cap_override) = self.effective_cap_override {
            effective_count = effective_count.min(cap_override.max(1_000));
        }
        effective_count
    }

    fn rebuild_point_state(&mut self, requested_count: usize, effective_count: usize) {
        self.requested_point_count = requested_count;
        self.point_count = effective_count;
        self.positions = generate_positions(self.point_count, self.max_radius);
        self.visibility.clear();
        self.visibility.resize(self.point_count, true);
        self.visible_indices.clear();
        self.visible_indices
            .extend((0..self.point_count).map(|i| i as u32));
        self.culled_points_count = 0;
        self.visibility_valid = false;
        self.points_initialized = true;
        self.max_intensity = 0.0;
        self.min_intensity = 0.0;
        self.avg_intensity = 0.0;
        self.zero_intensity_count = self.point_count;

        debug!(
            "Point count changed to requested={} effective={}",
            self.requested_point_count, self.point_count
        );
    }

    // Get positions of all points
    pub fn get_positions(&self) -> Vec<Vec3> {
        self.positions.clone()
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    pub fn effective_point_count(&self) -> usize {
        self.point_count
    }

    pub fn requested_point_count(&self) -> usize {
        self.requested_point_count
    }

    pub fn prioritized_compute_ranges(
        &self,
        block_size: usize,
        scheduled_points: usize,
    ) -> Vec<(usize, usize)> {
        let total_points = scheduled_points.min(self.point_count);
        if total_points == 0 {
            return Vec::new();
        }

        let block_size = block_size.max(1);
        if !self.visibility_valid || self.culled_points_count == 0 {
            return self.sequential_compute_ranges(block_size, total_points);
        }

        struct BlockPriority {
            start: usize,
            len: usize,
            score: f32,
        }

        let mut block_priorities = Vec::new();
        let mut start = 0usize;
        while start < self.point_count {
            let end = (start + block_size).min(self.point_count);
            let len = end - start;
            let sample_points = len.clamp(1, Self::COARSE_BLOCK_SAMPLE_POINTS);
            let step = (len / sample_points).max(1);

            let mut sampled = 0usize;
            let mut visible = 0usize;
            let mut index = start;
            while index < end && sampled < sample_points {
                if self.visibility[index] {
                    visible += 1;
                }
                sampled += 1;
                index = index.saturating_add(step);
            }

            let score = if sampled > 0 {
                visible as f32 / sampled as f32
            } else {
                0.0
            };
            block_priorities.push(BlockPriority { start, len, score });
            start = end;
        }

        block_priorities.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.start.cmp(&b.start))
        });

        let mut remaining = total_points;
        let mut ranges = Vec::new();
        for block in block_priorities {
            if remaining == 0 {
                break;
            }
            let take = block.len.min(remaining);
            ranges.push((block.start, take));
            remaining -= take;
        }
        ranges
    }

    fn sequential_compute_ranges(
        &self,
        block_size: usize,
        total_points: usize,
    ) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut remaining = total_points;
        let mut start = 0usize;

        while start < self.point_count && remaining > 0 {
            let len = block_size.min(self.point_count - start).min(remaining);
            ranges.push((start, len));
            start = start.saturating_add(block_size);
            remaining -= len;
        }
        ranges
    }

    // Update intensities directly (used with GPU acceleration)
    pub fn update_intensities(&mut self, intensities: &[f32]) {
        if intensities.len() != self.point_count {
            warn!(
                "Intensity array size mismatch: {} vs {}",
                intensities.len(),
                self.point_count
            );
            return;
        }

        RECALCULATIONS_COUNT.fetch_add(1, Ordering::Relaxed);

        if self.point_count == 0 {
            self.max_intensity = 0.0;
            self.min_intensity = 0.0;
            self.avg_intensity = 0.0;
            self.zero_intensity_count = 0;
            return;
        }

        let sampled_points = self.point_count.clamp(1, Self::INTENSITY_STATS_SAMPLE_POINTS);
        let step = (self.point_count / sampled_points).max(1);

        let mut max_intensity = 0.0f32;
        let mut sampled = 0usize;
        let mut index = 0usize;
        while index < self.point_count && sampled < sampled_points {
            let intensity = intensities[index];
            if intensity.is_finite() && !intensity.is_nan() {
                max_intensity = max_intensity.max(intensity);
            }
            sampled += 1;
            index = index.saturating_add(step);
        }
        max_intensity = max_intensity.max(0.001);

        let mut normalized_max_intensity = 0.0f32;
        let mut normalized_min_intensity = f32::INFINITY;
        let mut normalized_sum_intensity = 0.0f32;
        let mut normalized_zero_count = 0usize;

        sampled = 0;
        index = 0;
        while index < self.point_count && sampled < sampled_points {
            let mut intensity = intensities[index];
            if !intensity.is_finite() || intensity.is_nan() {
                intensity = 1.0;
            } else {
                intensity = (intensity / max_intensity).clamp(0.0, 1.0);
            }

            normalized_max_intensity = normalized_max_intensity.max(intensity);
            normalized_min_intensity = normalized_min_intensity.min(intensity);
            normalized_sum_intensity += intensity;
            if intensity < 0.000001 {
                normalized_zero_count += 1;
            }
            sampled += 1;
            index = index.saturating_add(step);
        }

        let sampled_count = sampled.max(1);
        self.max_intensity = normalized_max_intensity;
        self.min_intensity = normalized_min_intensity;
        self.avg_intensity = normalized_sum_intensity / sampled_count as f32;
        let zero_ratio = normalized_zero_count as f32 / sampled_count as f32;
        self.zero_intensity_count = (zero_ratio * self.point_count as f32)
            .round()
            .clamp(0.0, self.point_count as f32) as usize;

        debug!(
            "Updated intensity stats (sampled={}): min={:.6}, max={:.6}, avg={:.6}, zeros≈{}",
            sampled_count,
            self.min_intensity,
            self.max_intensity,
            self.avg_intensity,
            self.zero_intensity_count
        );
    }

    #[allow(dead_code)]
    pub fn set_frustum_culling(&mut self, enabled: bool) {
        self.use_frustum_culling = enabled;
    }
}

// Debug information structure
#[derive(Debug, Clone, Copy)]
pub struct PointCloudDebugInfo {
    pub point_count: usize,
    pub effective_point_count: usize,
    pub max_intensity: f32,
    pub min_intensity: f32,
    pub avg_intensity: f32,
    pub zero_intensity_count: usize,
    pub max_radius: f32,
    pub points_initialized: bool,
    pub cached_updates_count: usize,
    pub recalculations_count: usize,
    pub culled_points_count: usize,
    pub culling_percentage: f32,
    pub frame_ms: f32,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    pub scheduler_block_size: usize,
    pub scheduler_quality: f32,
    pub queue_depth: usize,
    pub stale_frames: u32,
    pub approx_mode: bool,
    pub cpu_worker_utilization: f32,
    pub native_input_points: usize,
    pub native_visible_points: usize,
    pub native_culled_points: usize,
    pub native_culling_percentage: f32,
    pub native_truncated_points: usize,
}

impl Default for PointCloudDebugInfo {
    fn default() -> Self {
        Self {
            point_count: 0,
            effective_point_count: 0,
            max_intensity: 0.0,
            min_intensity: 0.0,
            avg_intensity: 0.0,
            zero_intensity_count: 0,
            max_radius: 0.0,
            points_initialized: false,
            cached_updates_count: 0,
            recalculations_count: 0,
            culled_points_count: 0,
            culling_percentage: 0.0,
            frame_ms: 0.0,
            gpu_ms: 0.0,
            cpu_ms: 0.0,
            scheduler_block_size: 0,
            scheduler_quality: 1.0,
            queue_depth: 0,
            stale_frames: 0,
            approx_mode: false,
            cpu_worker_utilization: 0.0,
            native_input_points: 0,
            native_visible_points: 0,
            native_culled_points: 0,
            native_culling_percentage: 0.0,
            native_truncated_points: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prioritized_ranges_fallback_to_sequential_when_visibility_invalid() {
        let mut cloud = PointCloud::new(1_000);
        cloud.point_count = 16;
        cloud.positions.resize(16, Vec3::ZERO);
        cloud.visibility.resize(16, true);
        cloud.culled_points_count = 4;
        cloud.visibility_valid = false;

        let ranges = cloud.prioritized_compute_ranges(8, 16);
        assert_eq!(ranges, vec![(0, 8), (8, 8)]);
    }

    #[test]
    fn prioritized_ranges_prefer_visible_blocks() {
        let mut cloud = PointCloud::new(1_000);
        cloud.point_count = 16;
        cloud.positions.resize(16, Vec3::ZERO);
        cloud.visibility = vec![
            false, false, false, false, false, false, false, false, true, true, true, true, true,
            true, true, true,
        ];
        cloud.culled_points_count = 8;
        cloud.visibility_valid = true;

        let ranges = cloud.prioritized_compute_ranges(8, 8);
        assert_eq!(ranges, vec![(8, 8)]);
    }

    #[test]
    fn effective_cap_override_reduces_effective_count() {
        let mut cloud = PointCloud::new(2_000_000);
        cloud.set_effective_cap_override(Some(250_000));
        assert_eq!(cloud.requested_point_count(), 2_000_000);
        assert_eq!(cloud.effective_point_count(), 250_000);
    }
}
