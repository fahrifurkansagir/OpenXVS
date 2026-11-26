/**
 * REL4 Immutable Non-Flight Simulation Watermark Guard.
 *
 * Scopes its `MutationObserver` strictly to:
 *   1. Direct `{ childList: true }` on `rootEl` (catches removal/re-parenting of the watermark node).
 *   2. `{ attributes: true, characterData: true, childList: true, subtree: true }` ONLY on
 *      `#studio-non-flight-watermark` itself (catches tampering with its text, class, style, or attributes).
 *
 * Telemetry updates inside HUD, canvas, drawer, and dialogs trigger ZERO observer wakeups.
 */

import { NON_FLIGHT_WATERMARK } from "./bridge_client.ts";

export interface ImmutableWatermarkHandle {
  readonly dispose: () => void;
  readonly getRestoreCount: () => number;
}

export function ensureImmutableNonFlightWatermark(
  rootEl: HTMLElement,
): ImmutableWatermarkHandle {
  let rootObserver: MutationObserver | null = null;
  let watermarkObserver: MutationObserver | null = null;
  let applying = false;
  let restoreCount = 0;

  const ensureElement = (): HTMLElement => {
    const existing = document.getElementById("studio-non-flight-watermark");
    if (existing && existing.parentElement === rootEl) {
      return existing;
    }
    const el = existing ?? document.createElement("div");
    el.id = "studio-non-flight-watermark";
    rootEl.appendChild(el);
    return el;
  };

  const applyWatermark = (fromMutation: boolean): void => {
    if (applying) return;
    applying = true;
    rootObserver?.disconnect();
    watermarkObserver?.disconnect();
    try {
      const el = ensureElement();
      let mutated = false;

      if (el.className !== "studio-non-flight-watermark") {
        el.className = "studio-non-flight-watermark";
        mutated = true;
      }
      if (el.textContent !== NON_FLIGHT_WATERMARK) {
        el.textContent = NON_FLIGHT_WATERMARK;
        mutated = true;
      }
      if (el.getAttribute("role") !== "status") {
        el.setAttribute("role", "status");
        mutated = true;
      }
      if (el.getAttribute("aria-live") !== "polite") {
        el.setAttribute("aria-live", "polite");
        mutated = true;
      }
      if (el.getAttribute("data-non-flight-sim-only") !== "true") {
        el.setAttribute("data-non-flight-sim-only", "true");
        mutated = true;
      }
      if (
        el.style.getPropertyValue("display") !== "flex" ||
        el.style.getPropertyPriority("display") !== "important"
      ) {
        el.style.setProperty("display", "flex", "important");
        mutated = true;
      }
      if (
        el.style.getPropertyValue("visibility") !== "visible" ||
        el.style.getPropertyPriority("visibility") !== "important"
      ) {
        el.style.setProperty("visibility", "visible", "important");
        mutated = true;
      }
      if (
        el.style.getPropertyValue("opacity") !== "1" ||
        el.style.getPropertyPriority("opacity") !== "important"
      ) {
        el.style.setProperty("opacity", "1", "important");
        mutated = true;
      }
      if (fromMutation && mutated) {
        restoreCount++;
      }

      if (rootObserver) {
        rootObserver.takeRecords();
        rootObserver.observe(rootEl, { childList: true });
      }
      if (watermarkObserver) {
        watermarkObserver.takeRecords();
        watermarkObserver.observe(el, {
          attributes: true,
          characterData: true,
          childList: true,
          subtree: true,
        });
      }
    } finally {
      applying = false;
    }
  };

  if (typeof MutationObserver !== "undefined") {
    rootObserver = new MutationObserver((records) => {
      // Only act if the watermark node itself was removed or detached from rootEl
      const wm = document.getElementById("studio-non-flight-watermark");
      if (!wm || wm.parentElement !== rootEl) {
        restoreCount++;
        applyWatermark(false);
        return;
      }
      for (const rec of records) {
        for (const removed of Array.from(rec.removedNodes)) {
          if (removed instanceof HTMLElement && removed.id === "studio-non-flight-watermark") {
            restoreCount++;
            applyWatermark(false);
            return;
          }
        }
      }
    });
    watermarkObserver = new MutationObserver(() => {
      applyWatermark(true);
    });
  }

  applyWatermark(false);

  return {
    dispose: () => {
      rootObserver?.disconnect();
      watermarkObserver?.disconnect();
    },
    getRestoreCount: () => restoreCount,
  };
}
