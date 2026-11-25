/* OpenXVS v2.0 — Wait-Free Power-of-Two SPSC Ring Buffer (`xvs-rt`, DR-06 / SP3 §3.1).
 *
 * Strictly zero heap allocation, alignas(64) false-sharing isolation, acquire/release ordering.
 * Concurrency contract:
 *   - `try_push` / `try_pop` form the strictly wait-free, race-free concurrent SPSC pair
 *     (producer exclusively writes `head_`, consumer exclusively writes `tail_`).
 *   - `push_overwrite_latest` advances `tail_` from the producer side and MUST ONLY be used
 *     in strictly single-threaded or consumer-quiesced contexts (never concurrently with `try_pop`).
 */

#ifndef XVS_RT_SPSC_RING_HPP
#define XVS_RT_SPSC_RING_HPP

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace xvs_rt {

template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity >= 2U, "SpscRing Capacity must be >= 2");
    static_assert((Capacity & (Capacity - 1U)) == 0U, "SpscRing Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "SpscRing element type must be trivially copyable POD");

public:
    SpscRing() noexcept = default;
    SpscRing(const SpscRing &) = delete;
    SpscRing &operator=(const SpscRing &) = delete;

    static constexpr std::size_t capacity() noexcept { return Capacity; }
    static constexpr std::size_t usable_capacity() noexcept { return Capacity - 1U; }

    /* Strictly wait-free concurrent SPSC producer enqueue (paired with `try_pop`). */
    bool try_push(const T &item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next_head = (head + 1U) & kMask;
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (next_head == tail) {
            return false;
        }
        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /* Push item; if full, drop oldest element and write latest (returns true if dropped).
     * THREADING INVARIANT: strictly single-thread / consumer-quiesced ONLY because advancing
     * `tail_` from the producer side is NOT race-free against a concurrent `try_pop` consumer.
     * Use `try_push` / `try_pop` whenever producer and consumer run on distinct threads. */
    bool push_overwrite_latest(const T &item) noexcept {
        bool dropped = false;
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next_head = (head + 1U) & kMask;
        std::size_t tail = tail_.load(std::memory_order_acquire);
        if (next_head == tail) {
            /* Advance tail by 1 (single-producer overwrite semantics when consumer lags) */
            const std::size_t next_tail = (tail + 1U) & kMask;
            tail_.compare_exchange_strong(tail, next_tail, std::memory_order_acq_rel, std::memory_order_acquire);
            dropped = true;
        }
        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return dropped;
    }

    bool try_pop(T &out_item) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (tail == head) {
            return false;
        }
        out_item = buffer_[tail];
        tail_.store((tail + 1U) & kMask, std::memory_order_release);
        return true;
    }

    std::size_t size_approx() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & kMask;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    void reset() noexcept {
        head_.store(0U, std::memory_order_release);
        tail_.store(0U, std::memory_order_release);
    }

private:
    static constexpr std::size_t kMask = Capacity - 1U;

    alignas(64) std::atomic<std::size_t> head_{0U};
    alignas(64) std::atomic<std::size_t> tail_{0U};
    alignas(64) std::array<T, Capacity> buffer_{};
};

}  // namespace xvs_rt

#endif  // XVS_RT_SPSC_RING_HPP
