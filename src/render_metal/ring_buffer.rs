#[cfg(target_os = "macos")]
use metal::{Buffer, BufferRef, Device, MTLResourceOptions};

#[cfg(target_os = "macos")]
struct SharedRingSlot {
    buffer: Buffer,
    frame_id: u64,
}

#[cfg(target_os = "macos")]
pub struct SharedBufferRing {
    slots: Vec<SharedRingSlot>,
    slot_bytes: usize,
    next_slot: usize,
    label_prefix: String,
}

#[cfg(target_os = "macos")]
impl SharedBufferRing {
    const MIN_SLOT_BYTES: usize = 256;

    pub fn new(
        device: &Device,
        slot_count: usize,
        requested_slot_bytes: usize,
        label_prefix: &str,
    ) -> Result<Self, String> {
        if slot_count == 0 {
            return Err("shared ring requires at least one slot".to_string());
        }

        let slot_bytes = Self::aligned_size(requested_slot_bytes);
        let mut slots = Vec::with_capacity(slot_count);
        for index in 0..slot_count {
            slots.push(Self::create_slot(device, slot_bytes, label_prefix, index));
        }

        Ok(Self {
            slots,
            slot_bytes,
            next_slot: 0,
            label_prefix: label_prefix.to_string(),
        })
    }

    pub fn ensure_capacity(&mut self, device: &Device, requested_slot_bytes: usize) -> bool {
        let target_bytes = Self::aligned_size(requested_slot_bytes);
        if target_bytes <= self.slot_bytes {
            return false;
        }

        for (index, slot) in self.slots.iter_mut().enumerate() {
            *slot = Self::create_slot(device, target_bytes, &self.label_prefix, index);
        }
        self.slot_bytes = target_bytes;
        self.next_slot = 0;
        true
    }

    pub fn acquire(&mut self, frame_id: u64) -> &BufferRef {
        let index = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        self.slots[index].frame_id = frame_id;
        self.slots[index].buffer.as_ref()
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    fn create_slot(
        device: &Device,
        slot_bytes: usize,
        prefix: &str,
        index: usize,
    ) -> SharedRingSlot {
        let buffer = device.new_buffer(slot_bytes as u64, MTLResourceOptions::StorageModeShared);
        buffer.set_label(&format!("{prefix}-slot-{index}"));
        SharedRingSlot {
            buffer,
            frame_id: 0,
        }
    }

    fn aligned_size(requested: usize) -> usize {
        let requested = requested.max(Self::MIN_SLOT_BYTES);
        let remainder = requested % Self::MIN_SLOT_BYTES;
        if remainder == 0 {
            requested
        } else {
            requested + (Self::MIN_SLOT_BYTES - remainder)
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct SharedBufferRing {
    slot_count: usize,
    slot_bytes: usize,
}

#[cfg(not(target_os = "macos"))]
impl SharedBufferRing {
    pub fn new(
        _device: &(),
        slot_count: usize,
        requested_slot_bytes: usize,
        _label_prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            slot_count,
            slot_bytes: requested_slot_bytes,
        })
    }

    pub fn ensure_capacity(&mut self, _device: &(), requested_slot_bytes: usize) -> bool {
        if requested_slot_bytes > self.slot_bytes {
            self.slot_bytes = requested_slot_bytes;
            true
        } else {
            false
        }
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
}
