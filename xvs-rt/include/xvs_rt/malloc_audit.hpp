/* OpenXVS v2.0 — Linker `-Wl,--wrap=malloc/calloc/realloc/free` Zero-Alloc Hot-Loop Trap (`DR-06`).
 *
 * Guarded by atomic g_rt_audit_armed:
 *   - Setup (`xvs_rta_create()`) runs strictly BEFORE arming.
 *   - Hot loop (`10,000` ticks) runs while armed -> MUST record 0 allocs & 0 bytes under -O0 and -O2.
 *   - Teardown (`xvs_rta_destroy()`, stdio/JSON) runs strictly AFTER disarming.
 */

#ifndef XVS_RT_MALLOC_AUDIT_HPP
#define XVS_RT_MALLOC_AUDIT_HPP

#pragma once

#include <cstddef>
#include <cstdint>

namespace xvs_rt {

struct MallocAuditStats {
    uint64_t alloc_count;
    uint64_t alloc_bytes;
    uint64_t free_count;
};

void malloc_audit_reset_and_arm() noexcept;
MallocAuditStats malloc_audit_disarm_and_read() noexcept;
bool malloc_audit_is_armed() noexcept;

/* Positive self-test proving the linker wrap trap is actually live (not elided). */
MallocAuditStats malloc_audit_run_positive_selftest() noexcept;

}  // namespace xvs_rt

#endif  // XVS_RT_MALLOC_AUDIT_HPP
