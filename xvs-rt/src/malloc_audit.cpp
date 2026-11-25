/* OpenXVS v2.0 — Linker `-Wl,--wrap=malloc/calloc/realloc/free` Zero-Alloc Audit (`DR-06`). */

#include "xvs_rt/malloc_audit.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

alignas(64) std::atomic<bool> g_rt_audit_armed{false};
alignas(64) std::atomic<uint64_t> g_alloc_count{0U};
alignas(64) std::atomic<uint64_t> g_alloc_bytes{0U};
alignas(64) std::atomic<uint64_t> g_free_count{0U};

}  // namespace

extern "C" {

void *__real_malloc(size_t size);
void *__real_calloc(size_t nmemb, size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);

void *__wrap_malloc(size_t size) {
    if (g_rt_audit_armed.load(std::memory_order_acquire)) {
        g_alloc_count.fetch_add(1U, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }
    return __real_malloc(size);
}

void *__wrap_calloc(size_t nmemb, size_t size) {
    if (g_rt_audit_armed.load(std::memory_order_acquire)) {
        g_alloc_count.fetch_add(1U, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(
            static_cast<uint64_t>(nmemb) * static_cast<uint64_t>(size),
            std::memory_order_relaxed);
    }
    return __real_calloc(nmemb, size);
}

void *__wrap_realloc(void *ptr, size_t size) {
    if (g_rt_audit_armed.load(std::memory_order_acquire)) {
        g_alloc_count.fetch_add(1U, std::memory_order_relaxed);
        g_alloc_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }
    return __real_realloc(ptr, size);
}

void __wrap_free(void *ptr) {
    if (ptr != nullptr && g_rt_audit_armed.load(std::memory_order_acquire)) {
        g_free_count.fetch_add(1U, std::memory_order_relaxed);
    }
    __real_free(ptr);
}

}  // extern "C"

void *operator new(std::size_t size) {
    void *p = std::malloc(size == 0U ? 1U : size);
    return p;
}

void *operator new[](std::size_t size) {
    void *p = std::malloc(size == 0U ? 1U : size);
    return p;
}

void operator delete(void *ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void *ptr) noexcept {
    std::free(ptr);
}

void operator delete(void *ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void *ptr, std::size_t) noexcept {
    std::free(ptr);
}

namespace xvs_rt {

namespace {

__attribute__((noinline)) void *selftest_alloc_sink(std::size_t n_bytes) noexcept {
    void *volatile p = std::malloc(n_bytes);
    if (p != nullptr) {
        volatile unsigned char *bytes = static_cast<volatile unsigned char *>(p);
        bytes[0] = 0x5AU;
    }
    return p;
}

__attribute__((noinline)) void selftest_free_sink(void *p) noexcept {
    std::free(p);
}

}  // namespace

void malloc_audit_reset_and_arm() noexcept {
    g_alloc_count.store(0U, std::memory_order_relaxed);
    g_alloc_bytes.store(0U, std::memory_order_relaxed);
    g_free_count.store(0U, std::memory_order_relaxed);
    g_rt_audit_armed.store(true, std::memory_order_release);
}

MallocAuditStats malloc_audit_disarm_and_read() noexcept {
    g_rt_audit_armed.store(false, std::memory_order_release);
    MallocAuditStats st{};
    st.alloc_count = g_alloc_count.load(std::memory_order_acquire);
    st.alloc_bytes = g_alloc_bytes.load(std::memory_order_acquire);
    st.free_count = g_free_count.load(std::memory_order_acquire);
    return st;
}

bool malloc_audit_is_armed() noexcept {
    return g_rt_audit_armed.load(std::memory_order_acquire);
}

MallocAuditStats malloc_audit_run_positive_selftest() noexcept {
    malloc_audit_reset_and_arm();
    void *p = selftest_alloc_sink(64U);
    selftest_free_sink(p);
    return malloc_audit_disarm_and_read();
}

}  // namespace xvs_rt
