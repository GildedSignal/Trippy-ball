// Buffer pool diagnostics.
//
// This module keeps a stable diagnostics surface for UI and scheduler
// introspection code.

use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct UsageStats {
    pub usage: &'static str,
    pub count: usize,
    pub memory: u64,
}

#[derive(Clone, Default)]
pub struct BufferPoolStats {
    pub total_buffers: usize,
    pub total_memory: u64,
    pub usage_stats: Vec<UsageStats>,
    pub created_count: usize,
    pub reused_count: usize,
}

pub struct BufferPool {
    stats: BufferPoolStats,
}

impl BufferPool {
    pub fn new(_max_pool_size: usize) -> Self {
        Self {
            stats: BufferPoolStats::default(),
        }
    }

    pub fn get_stats(&self) -> BufferPoolStats {
        self.stats.clone()
    }
}

pub struct GlobalBufferPool {
    inner: Mutex<BufferPool>,
}

impl GlobalBufferPool {
    pub fn new(max_pool_size: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(BufferPool::new(max_pool_size)),
        })
    }

    pub fn get_stats(&self) -> BufferPoolStats {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_creation() {
        let pool = BufferPool::new(10);
        assert_eq!(pool.get_stats().created_count, 0);
        assert_eq!(pool.get_stats().reused_count, 0);
    }
}
